import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";


/* =========================================================
   CONFIGURAÇÃO
   ========================================================= */

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";

const TRANSFORMERS_URL =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MAX_NEW_TOKENS = 24;

const MAX_STORED_MESSAGES = 200;


/* =========================================================
   PERSONALIDADE
   ========================================================= */

const PERSONALITY = `
Você é Luna, uma inteligência artificial.

Você conversa em português brasileiro.

Responda de forma:
- natural
- direta
- amigável
- breve

Use palavras normais.

Não invente palavras.
Não repita frases.
Não copie a pergunta do usuário.

Se não souber algo, diga que não sabe.

Responda somente à mensagem do usuário.
`;


/* =========================================================
   ELEMENTOS
   ========================================================= */

const chat =
    document.getElementById("chat");

const input =
    document.getElementById("messageInput");

const sendButton =
    document.getElementById("sendButton");

const clearButton =
    document.getElementById("clearButton");

const status =
    document.getElementById("status");


/* =========================================================
   ESTADO
   ========================================================= */

let history =
    loadHistory();

let worker =
    null;

let workerUrl =
    null;

let workerReady =
    false;

let loading =
    false;

let generating =
    false;

let requestId =
    0;

let pending =
    new Map();


/* =========================================================
   HISTÓRICO
   ========================================================= */

function loadHistory() {

    try {

        const value =
            JSON.parse(
                localStorage.getItem(
                    "luna_history"
                ) || "[]"
            );

        if (
            Array.isArray(value)
        ) {

            return value;
        }

    } catch (error) {
        // Ignora histórico inválido.
    }

    return [];
}


function saveHistory() {

    if (
        history.length >
        MAX_STORED_MESSAGES
    ) {

        history =
            history.slice(
                -MAX_STORED_MESSAGES
            );
    }

    try {

        localStorage.setItem(
            "luna_history",
            JSON.stringify(
                history
            )
        );

    } catch (error) {
        // Histórico não é essencial para a geração.
    }
}


/* =========================================================
   STATUS
   ========================================================= */

function setStatus(
    text
) {

    status.textContent =
        text;
}


/* =========================================================
   ERRO
   ========================================================= */

function errorText(
    error
) {

    if (
        error &&
        error.message
    ) {

        return error.message;
    }

    return String(
        error ||
        "Erro desconhecido"
    );
}


/* =========================================================
   MENSAGEM
   ========================================================= */

function addMessage(
    text,
    type
) {

    const element =
        document.createElement(
            "div"
        );

    element.className =
        `message ${type}`;

    element.textContent =
        String(
            text || ""
        );

    chat.appendChild(
        element
    );

    chat.scrollTop =
        chat.scrollHeight;

    return element;
}


/* =========================================================
   CARREGAR CONVERSA NA TELA
   ========================================================= */

function loadHistoryOnScreen() {

    for (
        const item of history
    ) {

        if (
            !item ||
            typeof item.content !==
                "string"
        ) {

            continue;
        }

        addMessage(
            item.content,
            item.role === "user"
                ? "user"
                : "ai"
        );
    }
}


/* =========================================================
   LIMPAR RESPOSTA
   ========================================================= */

function cleanAnswer(
    text
) {

    return String(
        text || ""
    )

        .replace(
            /^assistant\s*:\s*/i,
            ""
        )

        .replace(
            /^luna\s*:\s*/i,
            ""
        )

        .replace(
            /<\|im_end\|>[\s\S]*$/g,
            ""
        )

        .replace(
            /<\|endoftext\|>[\s\S]*$/g,
            ""
        )

        .trim();
}


/* =========================================================
   EXTRAIR RESPOSTA
   ========================================================= */

function extractAnswer(
    output
) {

    if (
        !output ||
        !Array.isArray(output) ||
        !output[0]
    ) {

        return "";
    }

    const generated =
        output[0].generated_text;


    if (
        Array.isArray(
            generated
        )
    ) {

        for (
            let i =
                generated.length - 1;

            i >= 0;

            i -= 1
        ) {

            const item =
                generated[i];

            if (
                item &&
                item.role ===
                    "assistant" &&
                typeof item.content ===
                    "string"
            ) {

                return item.content;
            }
        }

        return "";
    }


    if (
        typeof generated ===
            "string"
    ) {

        return generated;
    }


    return "";
}


/* =========================================================
   BLOQUEAR/DESBLOQUEAR INTERFACE
   ========================================================= */

function setBusy(
    value
) {

    generating =
        value;

    sendButton.disabled =
        value;

    input.disabled =
        value;
}


/* =========================================================
   WEB WORKER
   ========================================================= */

/*
 * O modelo roda dentro do Worker.
 *
 * A página principal continua responsável
 * somente pela interface.
 *
 * IMPORTANTE:
 *
 * Nesta versão:
 *
 * - sem histórico no modelo
 * - sem memória no modelo
 * - sem cache KV
 * - somente uma mensagem
 */

const workerSource = `

    import {
        pipeline,
        env
    } from "${TRANSFORMERS_URL}";


    /* -----------------------------------------
       WASM
       ----------------------------------------- */

    env.backends.onnx.wasm.numThreads = 1;

    env.backends.onnx.wasm.proxy = false;


    /* -----------------------------------------
       ESTADO DO MODELO
       ----------------------------------------- */

    let generator = null;

    let loading = false;

    let generating = false;


    /* -----------------------------------------
       MENSAGENS DO WORKER
       ----------------------------------------- */

    self.onmessage =
        async function(event) {

            const message =
                event.data || {};


            /* =================================
               CARREGAR MODELO
               ================================= */

            if (
                message.type ===
                "load"
            ) {

                if (
                    loading ||
                    generator
                ) {

                    return;
                }


                loading =
                    true;


                self.postMessage({
                    type:
                        "load-start"
                });


                try {

                    generator =
                        await pipeline(
                            "text-generation",

                            message.model,

                            {
                                device:
                                    "wasm",

                                dtype:
                                    "q4",

                                progress_callback:
                                    function(progress) {

                                        if (
                                            progress
                                        ) {

                                            self.postMessage({
                                                type:
                                                    "progress",

                                                progress:
                                                    progress
                                            });
                                        }
                                    }
                            }
                        );


                    self.postMessage({
                        type:
                            "ready"
                    });


                } catch (
                    error
                ) {

                    self.postMessage({
                        type:
                            "load-error",

                        error:
                            error &&
                            error.message
                                ? error.message
                                : String(error)
                    });


                } finally {

                    loading =
                        false;
                }


                return;
            }


            /* =================================
               GERAÇÃO
               ================================= */

            if (
                message.type !==
                "generate"
            ) {

                return;
            }


            if (
                !generator
            ) {

                self.postMessage({
                    type:
                        "generation-error",

                    id:
                        message.id,

                    error:
                        "Modelo ainda não está carregado."
                });

                return;
            }


            /*
             * Nunca permite duas gerações
             * simultâneas dentro do worker.
             */

            if (
                generating
            ) {

                self.postMessage({
                    type:
                        "generation-error",

                    id:
                        message.id,

                    error:
                        "Outra geração já está em andamento."
                });

                return;
            }


            generating =
                true;


            try {

                /*
                 * Apenas uma mensagem é enviada.
                 *
                 * Não existe histórico.
                 * Não existe memória.
                 */

                const output =
                    await generator(

                        message.messages,

                        {

                            /*
                             * Resposta curta.
                             */

                            max_new_tokens:
                                message.maxNewTokens,


                            /*
                             * Geração determinística.
                             */

                            do_sample:
                                false,


                            /*
                             * Não repetir
                             * o prompt na saída.
                             */

                            return_full_text:
                                false,


                            /*
                             * TESTE IMPORTANTE:
                             *
                             * desativamos o KV cache.
                             */

                            use_cache:
                                false
                        }
                    );


                /*
                 * Envia o resultado para
                 * a página principal.
                 */

                self.postMessage({

                    type:
                        "result",

                    id:
                        message.id,

                    output:
                        output
                });


            } catch (
                error
            ) {

                self.postMessage({

                    type:
                        "generation-error",

                    id:
                        message.id,

                    error:
                        error &&
                        error.message
                            ? error.message
                            : String(error)
                });


            } finally {

                generating =
                    false;
            }
        };
`;


/* =========================================================
   CRIAR WORKER
   ========================================================= */

function createWorker() {

    if (
        worker
    ) {

        return worker;
    }


    const blob =
        new Blob(
            [workerSource],
            {
                type:
                    "text/javascript"
            }
        );


    workerUrl =
        URL.createObjectURL(
            blob
        );


    worker =
        new Worker(
            workerUrl,
            {
                type:
                    "module"
            }
        );


    worker.addEventListener(
        "message",
        handleWorkerMessage
    );


    worker.addEventListener(
        "error",
        handleWorkerError
    );


    return worker;
}


/* =========================================================
   MENSAGENS DO WORKER
   ========================================================= */

function handleWorkerMessage(
    event
) {

    const message =
        event.data || {};


    /* -----------------------------------------
       DOWNLOAD
       ----------------------------------------- */

    if (
        message.type ===
        "load-start"
    ) {

        setStatus(
            "Carregando Luna..."
        );

        return;
    }


    /* -----------------------------------------
       PROGRESSO
       ----------------------------------------- */

    if (
        message.type ===
        "progress"
    ) {

        const progress =
            message.progress ||
            {};


        if (
            progress.status ===
            "progress"
        ) {

            const value =
                Number(
                    progress.progress
                );


            if (
                Number.isFinite(
                    value
                )
            ) {

                setStatus(
                    `Baixando IA... ${Math.round(value)}%`
                );

            } else {

                setStatus(
                    "Baixando IA..."
                );
            }


        } else if (
            progress.status ===
            "done"
        ) {

            setStatus(
                "Finalizando..."
            );
        }


        return;
    }


    /* -----------------------------------------
       MODELO PRONTO
       ----------------------------------------- */

    if (
        message.type ===
        "ready"
    ) {

        loading =
            false;

        workerReady =
            true;

        setStatus(
            "Online - worker local"
        );

        addMessage(
            "Luna está online.",
            "ai"
        );

        return;
    }


    /* -----------------------------------------
       ERRO NO MODELO
       ----------------------------------------- */

    if (
        message.type ===
        "load-error"
    ) {

        loading =
            false;

        workerReady =
            false;

        setStatus(
            "Erro ao carregar IA"
        );

        addMessage(
            `ERRO AO CARREGAR:\n\n${message.error}`,
            "ai"
        );

        return;
    }


    /* -----------------------------------------
       RESULTADO
       ----------------------------------------- */

    if (
        message.type ===
        "result"
    ) {

        const item =
            pending.get(
                message.id
            );


        if (!item) {
            return;
        }


        pending.delete(
            message.id
        );


        item.resolve(
            message.output
        );

        return;
    }


    /* -----------------------------------------
       ERRO NA GERAÇÃO
       ----------------------------------------- */

    if (
        message.type ===
        "generation-error"
    ) {

        const item =
            pending.get(
                message.id
            );


        if (!item) {
            return;
        }


        pending.delete(
            message.id
        );


        item.reject(
            new Error(
                message.error
            )
        );
    }
}


/* =========================================================
   ERRO DO WORKER
   ========================================================= */

function handleWorkerError(
    event
) {

    loading =
        false;

    workerReady =
        false;


    for (
        const item of
        pending.values()
    ) {

        item.reject(
            new Error(
                "O worker de inferência foi encerrado."
            )
        );
    }


    pending.clear();


    if (
        generating
    ) {

        setBusy(
            false
        );

        addMessage(
            "A geração foi interrompida porque o worker falhou.",
            "ai"
        );
    }


    setStatus(
        "Worker indisponível"
    );
}


/* =========================================================
   PEDIR GERAÇÃO
   ========================================================= */

function requestGeneration(
    messages
) {

    return new Promise(
        (
            resolve,
            reject
        ) => {

            const id =
                ++requestId;


            pending.set(
                id,
                {
                    resolve,
                    reject
                }
            );


            worker.postMessage({

                type:
                    "generate",

                id:
                    id,

                messages:
                    messages,

                maxNewTokens:
                    MAX_NEW_TOKENS
            });
        }
    );
}


/* =========================================================
   GERAR
   ========================================================= */

async function generate(
    userMessage
) {

    if (
        !workerReady ||
        generating
    ) {

        return;
    }


    setBusy(
        true
    );


    /*
     * Salva no histórico somente
     * para manter a conversa na página.
     *
     * NÃO será enviado para o modelo.
     */

    history.push({

        role:
            "user",

        content:
            userMessage
    });


    saveHistory();


    const thinking =
        addMessage(
            "Pensando...",
            "ai"
        );


    try {

        /*
         * TESTE ISOLADO:
         *
         * somente a mensagem atual.
         */

        const messages = [

            {
                role:
                    "system",

                content:
                    PERSONALITY
            },

            {
                role:
                    "user",

                content:
                    userMessage
            }

        ];


        const output =
            await requestGeneration(
                messages
            );


        const answer =
            cleanAnswer(
                extractAnswer(
                    output
                )
            );


        thinking.remove();


        if (
            answer
        ) {

            addMessage(
                answer,
                "ai"
            );


            history.push({

                role:
                    "assistant",

                content:
                    answer
            });


            saveHistory();

        } else {

            addMessage(
                "Não consegui gerar uma resposta.",
                "ai"
            );
        }


    } catch (
        error
    ) {

        thinking.remove();


        addMessage(
            `ERRO AO RESPONDER:\n\n${errorText(error)}`,
            "ai"
        );


    } finally {

        setBusy(
            false
        );

        input.focus();
    }
}


/* =========================================================
   ENVIAR
   ========================================================= */

async function handleSend() {

    if (
        loading ||
        generating ||
        !workerReady
    ) {

        return;
    }


    const message =
        input.value.trim();


    if (!message) {
        return;
    }


    /*
     * Limpa imediatamente.
     */

    input.value =
        "";


    /*
     * Mostra a mensagem
     * imediatamente.
     */

    addMessage(
        message,
        "user"
    );


    /*
     * Dá uma pequena chance
     * para o navegador atualizar
     * a interface antes da geração.
     */

    await new Promise(
        resolve =>
            setTimeout(
                resolve,
                0
            )
    );


    await generate(
        message
    );
}


/* =========================================================
   BOTÃO
   ========================================================= */

sendButton.addEventListener(
    "click",
    handleSend
);


/* =========================================================
   ENTER
   ========================================================= */

input.addEventListener(
    "keydown",
    event => {

        if (
            event.key ===
                "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();


            if (
                !loading &&
                !generating
            ) {

                handleSend();
            }
        }
    }
);


/* =========================================================
   LIMPAR
   ========================================================= */

clearButton.addEventListener(
    "click",
    () => {

        if (
            loading ||
            generating
        ) {

            return;
        }


        history =
            [];

        saveHistory();


        chat.innerHTML =
            "";


        setStatus(
            workerReady
                ? "Online - worker local"
                : "Modelo não carregado"
        );
    }
);


/* =========================================================
   INICIAR
   ========================================================= */

async function start() {

    loadHistoryOnScreen();


    loading =
        true;


    setStatus(
        "Iniciando worker..."
    );


    const inferenceWorker =
        createWorker();


    inferenceWorker.postMessage({

        type:
            "load",

        model:
            MODEL
    });
}


start();

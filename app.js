import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "PengZhang424242/SmolLM2-360M-Instruct-ONNX";

const PERSONALITY = `
Você é uma inteligência artificial chamada Luna.

Você conversa em português brasileiro.

Sua personalidade:

- natural
- inteligente
- direta
- amigável
- curiosa
- paciente

Você não é uma pessoa humana.

Não invente informações.

Se não souber algo, diga que não sabe.

Responda de maneira natural e fácil de entender.
`;

let generator = null;
let generating = false;

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


/* =========================
   MEMÓRIA
========================= */

let memory = JSON.parse(
    localStorage.getItem("luna_memory") || "{}"
);

let history = JSON.parse(
    localStorage.getItem("luna_history") || "[]"
);


function saveMemory() {
    localStorage.setItem(
        "luna_memory",
        JSON.stringify(memory)
    );
}


function saveHistory() {
    localStorage.setItem(
        "luna_history",
        JSON.stringify(history)
    );
}


/* =========================
   CHAT
========================= */

function addMessage(
    text,
    type
) {
    const element =
        document.createElement("div");

    element.className =
        "message " + type;

    element.textContent =
        text;

    chat.appendChild(element);

    chat.scrollTop =
        chat.scrollHeight;

    return element;
}


/* =========================
   CARREGAR HISTÓRICO
========================= */

function loadHistory() {

    for (
        const message
        of history
    ) {

        addMessage(
            message.content,
            message.role === "user"
                ? "user"
                : "ai"
        );
    }
}


/* =========================
   CARREGAR MODELO
========================= */

async function loadModel() {

    status.textContent =
        "Preparando IA...";

    try {

        let device;

        /*
         * Se o navegador tiver WebGPU,
         * tenta usar a GPU.
         *
         * Caso contrário,
         * usa WASM/CPU.
         */

        if (
            typeof navigator !== "undefined" &&
            navigator.gpu
        ) {

            device = "webgpu";

            status.textContent =
                "Iniciando IA pela GPU...";

        } else {

            device = "wasm";

            status.textContent =
                "Iniciando IA pela CPU...";

        }


        generator =
            await pipeline(
                "text-generation",
                MODEL,
                {
                    device: device,
                    dtype: "q4",

                    progress_callback:
                        (progress) => {

                            if (
                                progress &&
                                progress.status === "progress"
                            ) {

                                const value =
                                    Number(
                                        progress.progress
                                    );

                                if (
                                    Number.isFinite(value)
                                ) {

                                    status.textContent =
                                        `Baixando IA... ${Math.round(value)}%`;

                                } else {

                                    status.textContent =
                                        "Baixando IA...";

                                }

                            }

                            else if (
                                progress &&
                                progress.status === "initiate"
                            ) {

                                status.textContent =
                                    "Iniciando download...";

                            }

                            else if (
                                progress &&
                                progress.status === "done"
                            ) {

                                status.textContent =
                                    "Finalizando IA...";

                            }
                        }
                }
            );


        status.textContent =
            device === "webgpu"
                ? "Online - GPU"
                : "Online - CPU";


    } catch (error) {

        console.error(
            "Erro ao carregar Luna:",
            error
        );

        generator = null;

        status.textContent =
            "Erro ao carregar IA";

        addMessage(
            "A Luna não conseguiu carregar a IA. Recarregue a página e tente novamente.",
            "ai"
        );
    }
}


/* =========================
   GERAR RESPOSTA
========================= */

async function generate(
    userMessage
) {

    if (
        !generator ||
        generating
    ) {
        return;
    }


    generating = true;

    sendButton.disabled = true;


    /*
     * Salva mensagem do usuário
     */

    history.push({
        role: "user",
        content: userMessage
    });

    saveHistory();


    /*
     * Memória
     */

    const memoryText =
        Object.keys(memory).length
            ? JSON.stringify(memory)
            : "Nenhuma memória";


    /*
     * Últimas mensagens
     */

    const recentHistory =
        history.slice(-12);


    /*
     * Mensagens enviadas para a IA
     */

    const messages = [

        {
            role: "system",

            content:
                PERSONALITY +
                `

Memória da Luna:

${memoryText}
`
        },

        ...recentHistory

    ];


    /*
     * Mostra que está pensando
     */

    const thinking =
        addMessage(
            "Pensando...",
            "ai"
        );


    try {

        const output =
            await generator(
                messages,
                {
                    max_new_tokens: 180,

                    temperature: 0.7,

                    do_sample: true
                }
            );


        let answer = "";


        /*
         * Transformers.js retorna
         * generated_text.
         */

        if (
            output &&
            output[0]
        ) {

            const generated =
                output[0].generated_text;


            /*
             * Modelos de chat
             * normalmente retornam
             * uma lista de mensagens.
             */

            if (
                Array.isArray(
                    generated
                )
            ) {

                const last =
                    generated[
                        generated.length - 1
                    ];


                if (
                    typeof last === "string"
                ) {

                    answer =
                        last;

                }

                else if (
                    last &&
                    typeof last.content === "string"
                ) {

                    answer =
                        last.content;
                }

            }

            /*
             * Caso o modelo retorne
             * texto diretamente.
             */

            else if (
                typeof generated === "string"
            ) {

                answer =
                    generated;
            }
        }


        /*
         * Se não conseguiu encontrar
         * a resposta.
         */

        if (
            !answer ||
            !answer.trim()
        ) {

            answer =
                "Não consegui gerar uma resposta.";
        }


        /*
         * Remove o "Pensando..."
         */

        thinking.remove();


        /*
         * Mostra resposta
         */

        addMessage(
            answer.trim(),
            "ai"
        );


        /*
         * Salva resposta
         */

        history.push({
            role: "assistant",
            content: answer.trim()
        });

        saveHistory();


    } catch (error) {

        console.error(
            "Erro ao gerar resposta:",
            error
        );


        thinking.remove();


        addMessage(
            "Deu erro enquanto eu tentava responder.",
            "ai"
        );
    }


    generating = false;

    sendButton.disabled = false;

    input.focus();
}


/* =========================
   BOTÃO ENVIAR
========================= */

sendButton.addEventListener(
    "click",
    async () => {

        const message =
            input.value.trim();


        if (!message) {
            return;
        }


        /*
         * Se a IA ainda não carregou,
         * não tenta gerar.
         */

        if (!generator) {

            addMessage(
                "A Luna ainda está carregando. Aguarde terminar.",
                "ai"
            );

            return;
        }


        input.value = "";


        addMessage(
            message,
            "user"
        );


        await generate(
            message
        );
    }
);


/* =========================
   ENTER PARA ENVIAR
========================= */

input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendButton.click();
        }
    }
);


/* =========================
   LIMPAR CONVERSA
========================= */

clearButton.addEventListener(
    "click",
    () => {

        history = [];

        saveHistory();

        chat.innerHTML = "";
    }
);


/* =========================
   INICIAR LUNA
========================= */

async function start() {

    /*
     * Primeiro mostra
     * o histórico antigo.
     */

    loadHistory();


    /*
     * Depois carrega a IA.
     */

    await loadModel();
}


start();

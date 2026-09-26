import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct";

/*
 * Configuração para Chrome Android.
 *
 * Uma thread reduz o pico de RAM e CPU.
 * O modelo é carregado somente uma vez.
 */
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;


/* =========================================================
   PERSONALIDADE DA LUNA
   ========================================================= */

const PERSONALITY = `
Você é Luna, uma inteligência artificial.

Você conversa em português brasileiro.

Sua personalidade:
- natural
- inteligente
- direta
- amigável
- curiosa
- paciente

Regras importantes:

- Responda sempre em português brasileiro.
- Use palavras normais.
- Use frases naturais e fáceis de entender.
- Se a pergunta for simples, responda de forma simples.
- Não repita palavras ou frases sem necessidade.
- Não escreva palavras inventadas.
- Não misture idiomas.
- Não copie a mensagem do usuário.
- Não invente informações.
- Se não souber algo, diga que não sabe.
- Não fale sobre seu funcionamento interno, a menos que o usuário pergunte.
- Você não é uma pessoa humana.
`;


/* =========================================================
   LIMITES PARA CELULAR
   ========================================================= */

/*
 * Antes:
 *
 * q8
 * 64 tokens
 * 6000 caracteres
 *
 * Agora:
 *
 * q4
 * 32 tokens
 * 2500 caracteres
 *
 * Isso reduz bastante o trabalho durante cada resposta.
 */

const MAX_GENERATED_TOKENS = 32;

const MAX_CONTEXT_CHARS = 2500;

const MAX_MEMORY_CHARS = 800;

const MAX_SAVED_HISTORY_ITEMS = 100;


/* =========================================================
   ESTADO
   ========================================================= */

let generator = null;

let modelLoading = false;

let generating = false;


/* =========================================================
   ELEMENTOS DA INTERFACE
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
   MEMÓRIA E HISTÓRICO
   ========================================================= */

let memory = readJson(
    "luna_memory",
    {}
);

let history = readJson(
    "luna_history",
    []
);

if (
    !memory ||
    typeof memory !== "object" ||
    Array.isArray(memory)
) {
    memory = {};
}

if (!Array.isArray(history)) {
    history = [];
}


/* =========================================================
   LEITURA SEGURA DO LOCALSTORAGE
   ========================================================= */

function readJson(key, fallback) {
    try {
        const value =
            JSON.parse(
                localStorage.getItem(key) ||
                JSON.stringify(fallback)
            );

        return value;

    } catch (error) {

        console.warn(
            `Dados locais inválidos em ${key}.`,
            error
        );

        return fallback;
    }
}


/* =========================================================
   STATUS
   ========================================================= */

function setStatus(text) {
    status.textContent = text;
}


/* =========================================================
   SALVAR MEMÓRIA
   ========================================================= */

function saveMemory() {
    try {

        localStorage.setItem(
            "luna_memory",
            JSON.stringify(memory)
        );

    } catch (error) {

        console.warn(
            "Não foi possível salvar a memória local.",
            error
        );
    }
}


/* =========================================================
   SALVAR HISTÓRICO
   ========================================================= */

function saveHistory() {

    /*
     * Mantém somente os últimos itens
     * para evitar crescimento infinito.
     */

    if (
        history.length >
        MAX_SAVED_HISTORY_ITEMS
    ) {

        history =
            history.slice(
                -MAX_SAVED_HISTORY_ITEMS
            );
    }

    try {

        localStorage.setItem(
            "luna_history",
            JSON.stringify(history)
        );

    } catch (error) {

        console.warn(
            "Não foi possível salvar o histórico local.",
            error
        );
    }
}


/* =========================================================
   ADICIONAR MENSAGEM NA TELA
   ========================================================= */

function addMessage(
    text,
    type
) {

    const element =
        document.createElement("div");

    element.className =
        `message ${type}`;

    element.textContent =
        String(text || "");

    chat.appendChild(
        element
    );

    chat.scrollTop =
        chat.scrollHeight;

    return element;
}


/* =========================================================
   CARREGAR HISTÓRICO
   ========================================================= */

function loadHistory() {

    for (
        const message of history
    ) {

        if (
            !message ||
            typeof message.content !==
                "string"
        ) {
            continue;
        }

        addMessage(
            message.content,
            message.role === "user"
                ? "user"
                : "ai"
        );
    }
}


/* =========================================================
   PROGRESSO DO DOWNLOAD
   ========================================================= */

function progressCallback(
    progress
) {

    if (!progress) {
        return;
    }

    if (
        progress.status ===
        "initiate"
    ) {

        setStatus(
            "Iniciando download..."
        );

        return;
    }

    if (
        progress.status ===
        "progress"
    ) {

        const value =
            Number(
                progress.progress
            );

        setStatus(
            Number.isFinite(value)
                ? `Baixando IA... ${Math.round(value)}%`
                : "Baixando IA..."
        );

        return;
    }

    if (
        progress.status ===
        "done"
    ) {

        setStatus(
            "Finalizando..."
        );
    }
}


/* =========================================================
   CARREGAR MODELO
   ========================================================= */

async function loadModel() {

    if (
        generator ||
        modelLoading
    ) {
        return;
    }

    modelLoading = true;

    setStatus(
        "Preparando IA..."
    );

    try {

        setStatus(
            "Iniciando Luna pela CPU..."
        );

        generator =
            await pipeline(
                "text-generation",
                MODEL,
                {
                    device: "wasm",

                    /*
                     * Q4 usa menos memória
                     * que Q8.
                     */
                    dtype: "q4",

                    progress_callback:
                        progressCallback
                }
            );

        setStatus(
            "Online - CPU WASM q4"
        );

        addMessage(
            "Luna está online.",
            "ai"
        );

    } catch (error) {

        console.error(
            "ERRO COMPLETO AO CARREGAR O MODELO:",
            error
        );

        generator = null;

        setStatus(
            "Erro ao carregar IA"
        );

        addMessage(
            `ERRO REAL:\n\n${getErrorText(error)}`,
            "ai"
        );

    } finally {

        modelLoading = false;
    }
}


/* =========================================================
   TEXTO DO ERRO
   ========================================================= */

function getErrorText(
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
   MEMÓRIA
   ========================================================= */

function getMemoryText() {

    let text;

    try {

        text =
            JSON.stringify(
                memory
            );

    } catch (error) {

        text = "{}";
    }

    if (
        !text ||
        text === "{}"
    ) {

        return "Nenhuma memória salva.";
    }

    return text.slice(
        0,
        MAX_MEMORY_CHARS
    );
}


/* =========================================================
   VALIDAR MENSAGENS
   ========================================================= */

function isValidChatMessage(
    message
) {

    return Boolean(
        message &&
        (
            message.role === "user" ||
            message.role === "assistant"
        ) &&
        typeof message.content ===
            "string" &&
        message.content.trim()
    );
}


/* =========================================================
   HISTÓRICO ENVIADO AO MODELO
   ========================================================= */

function getPromptHistory() {

    const selected = [];

    let usedChars = 0;

    /*
     * Começa pelas mensagens mais recentes.
     */

    for (
        let index =
            history.length - 1;

        index >= 0;

        index -= 1
    ) {

        const message =
            history[index];

        if (
            !isValidChatMessage(
                message
            )
        ) {
            continue;
        }

        const content =
            message.content.trim();

        const cost =
            content.length + 40;

        if (
            selected.length > 0 &&
            usedChars + cost >
                MAX_CONTEXT_CHARS
        ) {
            break;
        }

        selected.push({
            role:
                message.role,

            content:
                content
        });

        usedChars += cost;
    }

    return selected.reverse();
}


/* =========================================================
   MONTAR MENSAGENS
   ========================================================= */

function buildMessages() {

    return [

        {
            role: "system",

            content:
                `${PERSONALITY}

Memória da Luna:

${getMemoryText()}

Responda apenas à última mensagem do usuário.`
        },

        ...getPromptHistory()
    ];
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
        output[0]
            .generated_text;

    if (
        Array.isArray(
            generated
        )
    ) {

        for (
            let index =
                generated.length - 1;

            index >= 0;

            index -= 1
        ) {

            const message =
                generated[index];

            if (
                message &&
                message.role ===
                    "assistant" &&
                typeof message.content ===
                    "string"
            ) {

                return message.content.trim();
            }
        }

        return "";
    }

    if (
        typeof generated ===
            "string"
    ) {

        return generated.trim();
    }

    return "";
}


/* =========================================================
   LIMPAR RESPOSTA
   ========================================================= */

function cleanAnswer(
    text
) {

    if (!text) {
        return "";
    }

    return String(text)

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
   GERAR RESPOSTA
   ========================================================= */

async function generate(
    userMessage
) {

    /*
     * Impede duas gerações
     * ao mesmo tempo.
     */

    if (
        !generator ||
        generating ||
        modelLoading
    ) {

        return;
    }

    generating = true;

    sendButton.disabled =
        true;

    input.disabled =
        true;


    /* -----------------------------------------
       SALVAR MENSAGEM DO USUÁRIO
       ----------------------------------------- */

    history.push({

        role:
            "user",

        content:
            userMessage
    });

    saveHistory();


    /* -----------------------------------------
       MENSAGEM TEMPORÁRIA
       ----------------------------------------- */

    const thinking =
        addMessage(
            "Pensando...",
            "ai"
        );

    let output = null;


    try {

        /*
         * O modelo continua carregado.
         * Não é carregado novamente
         * a cada mensagem.
         */

        output =
            await generator(
                buildMessages(),
                {

                    /*
                     * Menos tokens =
                     * menos processamento.
                     */
                    max_new_tokens:
                        MAX_GENERATED_TOKENS,

                    /*
                     * Desativa amostragem
                     * para reduzir variação
                     * e processamento.
                     */
                    do_sample:
                        false,

                    /*
                     * Não devolve novamente
                     * todo o texto de entrada.
                     */
                    return_full_text:
                        false,

                    /*
                     * Tokens de parada.
                     */
                    eos_token_id:
                        [
                            151645,
                            151643
                        ],

                    pad_token_id:
                        151643
                }
            );


        /* -------------------------------------
           PEGAR RESPOSTA
           ------------------------------------- */

        const answer =
            cleanAnswer(
                extractAnswer(
                    output
                )
            ) ||
            "Não consegui gerar uma resposta.";


        /* -------------------------------------
           MOSTRAR RESPOSTA
           ------------------------------------- */

        thinking.remove();

        addMessage(
            answer,
            "ai"
        );


        /* -------------------------------------
           SALVAR RESPOSTA
           ------------------------------------- */

        history.push({

            role:
                "assistant",

            content:
                answer
        });

        saveHistory();


    } catch (error) {

        console.error(
            "ERRO AO RESPONDER:",
            error
        );

        thinking.remove();

        addMessage(
            `ERRO AO RESPONDER:\n\n${getErrorText(error)}`,
            "ai"
        );

    } finally {

        /*
         * Libera a referência ao resultado
         * da geração.
         *
         * NÃO usamos dispose() no modelo,
         * porque isso faria o modelo precisar
         * ser carregado novamente.
         */

        output = null;

        generating =
            false;

        sendButton.disabled =
            false;

        input.disabled =
            false;

        input.focus();
    }
}


/* =========================================================
   BOTÃO ENVIAR
   ========================================================= */

sendButton.addEventListener(
    "click",
    async () => {

        const message =
            input.value.trim();

        if (
            !message ||
            generating ||
            modelLoading
        ) {

            return;
        }

        if (!generator) {

            addMessage(
                "A Luna ainda não terminou de carregar.",
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


/* =========================================================
   ENTER PARA ENVIAR
   ========================================================= */

input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !generating
        ) {

            event.preventDefault();

            sendButton.click();
        }
    }
);


/* =========================================================
   LIMPAR CONVERSA
   ========================================================= */

clearButton.addEventListener(
    "click",
    () => {

        if (generating) {
            return;
        }

        history = [];

        saveHistory();

        chat.innerHTML = "";

        setStatus(
            generator
                ? "Online - CPU WASM q4"
                : "Modelo não carregado"
        );
    }
);


/* =========================================================
   INICIAR LUNA
   ========================================================= */

async function start() {

    loadHistory();

    await loadModel();
}

start();

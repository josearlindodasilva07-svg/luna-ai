import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct";

/*
 * Configuração conservadora para Android.
 *
 * q8 é usado deliberadamente para evitar possíveis erros
 * de qualidade/inferência do model_q4.onnx em determinados
 * runtimes WASM do Chrome Android.
 */
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

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

Responda sempre em português brasileiro.

Use palavras normais e frases naturais.

Se o usuário fizer uma pergunta simples,
responda de forma simples.

Não repita palavras ou frases sem necessidade.

Não escreva palavras inventadas.

Não misture idiomas.

Não copie a mensagem do usuário.

Não invente informações.

Se não souber algo, diga que não sabe.

Não fale sobre seu funcionamento interno,
a menos que o usuário pergunte.

Você não é uma pessoa humana.
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

let memory = {};

let history = [];

try {
    memory = JSON.parse(
        localStorage.getItem("luna_memory") || "{}"
    );

    if (
        !memory ||
        typeof memory !== "object" ||
        Array.isArray(memory)
    ) {
        memory = {};
    }
} catch (error) {
    console.warn(
        "Memória local inválida. Ela será reiniciada.",
        error
    );

    memory = {};
}

try {
    history = JSON.parse(
        localStorage.getItem("luna_history") || "[]"
    );

    if (!Array.isArray(history)) {
        history = [];
    }
} catch (error) {
    console.warn(
        "Histórico local inválido. Ele será reiniciado.",
        error
    );

    history = [];
}

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

function setStatus(text) {
    status.textContent = text;
}

function addMessage(text, type) {
    const element =
        document.createElement("div");

    element.className =
        "message " + type;

    element.textContent =
        String(text || "");

    chat.appendChild(element);

    chat.scrollTop =
        chat.scrollHeight;

    return element;
}

function loadHistory() {
    for (const message of history) {
        if (
            !message ||
            typeof message.content !== "string"
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

function progressCallback(progress) {
    if (!progress) {
        return;
    }

    if (progress.status === "initiate") {
        setStatus(
            "Iniciando download..."
        );

        return;
    }

    if (progress.status === "progress") {
        const value =
            Number(progress.progress);

        if (Number.isFinite(value)) {
            setStatus(
                "Baixando IA... " +
                Math.round(value) +
                "%"
            );
        } else {
            setStatus(
                "Baixando IA..."
            );
        }

        return;
    }

    if (progress.status === "done") {
        setStatus(
            "Finalizando..."
        );

        return;
    }

    if (progress.status === "ready") {
        setStatus(
            "Preparando inferência..."
        );
    }
}

async function loadModel() {
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
                     * q8 é intencional.
                     *
                     * Use q4 somente depois de confirmar que
                     * q8 produz texto normal neste aparelho.
                     */
                    dtype: "q8",

                    progress_callback:
                        progressCallback
                }
            );

        setStatus(
            "Online - CPU WASM q8"
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

        const errorText =
            error && error.message
                ? error.message
                : String(error || "Erro desconhecido");

        addMessage(
            "ERRO REAL:\n\n" +
            errorText,
            "ai"
        );
    }
}

function extractAnswer(output) {
    if (
        !output ||
        !Array.isArray(output) ||
        !output[0]
    ) {
        return "";
    }

    const generated =
        output[0].generated_text;

    /*
     * Quando a entrada é uma lista de mensagens,
     * o Transformers.js retorna um Chat:
     *
     * [
     *   { role: "system", content: "..." },
     *   { role: "user", content: "..." },
     *   { role: "assistant", content: "..." }
     * ]
     */
    if (Array.isArray(generated)) {
        const assistantMessages =
            generated.filter(
                message =>
                    message &&
                    message.role === "assistant" &&
                    typeof message.content === "string"
            );

        if (assistantMessages.length > 0) {
            return assistantMessages[
                assistantMessages.length - 1
            ].content.trim();
        }

        const last =
            generated[generated.length - 1];

        if (
            last &&
            typeof last.content === "string"
        ) {
            return last.content.trim();
        }

        if (
            typeof last === "string"
        ) {
            return last.trim();
        }

        return "";
    }

    if (typeof generated === "string") {
        return generated.trim();
    }

    return "";
}

function cleanAnswer(text) {
    if (!text) {
        return "";
    }

    let answer =
        String(text).trim();

    answer =
        answer.replace(
            /^assistant\s*:\s*/i,
            ""
        );

    answer =
        answer.replace(
            /^luna\s*:\s*/i,
            ""
        );

    answer =
        answer.replace(
            /<\|im_end\|>[\s\S]*$/g,
            ""
        );

    answer =
        answer.replace(
            /<\|endoftext\|>[\s\S]*$/g,
            ""
        );

    return answer.trim();
}

function buildMessages() {
    const memoryText =
        Object.keys(memory).length > 0
            ? JSON.stringify(memory)
            : "Nenhuma memória salva.";

    /*
     * O histórico já contém a mensagem atual do usuário.
     * Limitar o contexto evita crescimento excessivo no Android.
     */
    const recentHistory =
        history
            .filter(
                message =>
                    message &&
                    (
                        message.role === "user" ||
                        message.role === "assistant"
                    ) &&
                    typeof message.content === "string"
            )
            .slice(-8);

    return [
        {
            role: "system",
            content:
                PERSONALITY +
                `

Memória da Luna:
${memoryText}

Agora responda apenas ao usuário.
`
        },
        ...recentHistory
    ];
}

async function generate(userMessage) {
    if (
        !generator ||
        generating
    ) {
        return;
    }

    generating = true;

    sendButton.disabled = true;
    input.disabled = true;

    history.push({
        role: "user",
        content: userMessage
    });

    saveHistory();

    const messages =
        buildMessages();

    const thinking =
        addMessage(
            "Pensando...",
            "ai"
        );

    try {
        /*
         * O template oficial do tokenizer Qwen é aplicado
         * automaticamente porque a entrada é uma lista de
         * mensagens.
         */
        const output =
            await generator(
                messages,
                {
                    max_new_tokens: 100,

                    /*
                     * Greedy decoding para diagnosticar
                     * modelo/tokenizer/backend sem ruído
                     * adicional de amostragem.
                     */
                    do_sample: false,

                    return_full_text: false,

                    eos_token_id: [
                        151645,
                        151643
                    ],

                    pad_token_id: 151643
                }
            );

        let answer =
            extractAnswer(output);

        answer =
            cleanAnswer(answer);

        if (!answer) {
            answer =
                "Não consegui gerar uma resposta.";
        }

        thinking.remove();

        addMessage(
            answer,
            "ai"
        );

        history.push({
            role: "assistant",
            content: answer
        });

        saveHistory();

    } catch (error) {
        console.error(
            "ERRO AO RESPONDER:",
            error
        );

        thinking.remove();

        const errorText =
            error && error.message
                ? error.message
                : String(error || "Erro desconhecido");

        addMessage(
            "ERRO AO RESPONDER:\n\n" +
            errorText,
            "ai"
        );

        /*
         * A mensagem do usuário permanece no histórico,
         * mas nenhuma resposta falsa é adicionada.
         */
    } finally {
        generating = false;

        sendButton.disabled = false;
        input.disabled = false;

        input.focus();
    }
}

sendButton.addEventListener(
    "click",
    async () => {
        const message =
            input.value.trim();

        if (!message) {
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

clearButton.addEventListener(
    "click",
    () => {
        history = [];

        saveHistory();

        chat.innerHTML = "";

        setStatus(
            generator
                ? "Online - CPU WASM q8"
                : "Modelo não carregado"
        );
    }
);

async function start() {
    loadHistory();

    await loadModel();
}

start();

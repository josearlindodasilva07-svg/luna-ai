import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA";

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

Regras:
- Responda em português brasileiro.
- Responda de forma natural.
- Use frases claras e fáceis de entender.
- Não repita palavras ou frases sem necessidade.
- Não invente informações.
- Se não souber algo, diga que não sabe.
- Não fique repetindo a pergunta do usuário.
- Não faça listas quando uma resposta normal for melhor.
- Não fale sobre seu funcionamento interno, a menos que o usuário pergunte.
- Você não é uma pessoa humana.
`;

let generator = null;
let generating = false;

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

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

function setStatus(text) {
    status.textContent = text;
}

function addMessage(text, type) {
    const element =
        document.createElement("div");

    element.className =
        "message " + type;

    element.textContent = text;

    chat.appendChild(element);
    chat.scrollTop = chat.scrollHeight;

    return element;
}

function loadHistory() {
    for (const message of history) {
        addMessage(
            message.content,
            message.role === "user"
                ? "user"
                : "ai"
        );
    }
}

function progressCallback(progress) {
    if (!progress) return;

    if (progress.status === "initiate") {
        setStatus("Iniciando download...");
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
            setStatus("Baixando IA...");
        }

        return;
    }

    if (progress.status === "done") {
        setStatus("Finalizando...");
    }
}

async function loadModel() {
    setStatus("Preparando IA...");

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
                    dtype: "q4",
                    progress_callback:
                        progressCallback
                }
            );

        setStatus("Online - CPU");

        addMessage(
            "Luna está online.",
            "ai"
        );

    } catch (error) {
        console.error(
            "ERRO COMPLETO:",
            error
        );

        generator = null;

        setStatus(
            "Erro ao carregar IA"
        );

        let errorText =
            "Erro desconhecido";

        if (
            error &&
            error.message
        ) {
            errorText =
                error.message;
        } else if (
            typeof error === "string"
        ) {
            errorText = error;
        }

        addMessage(
            "ERRO REAL:\n\n" +
            errorText,
            "ai"
        );
    }
}

function cleanAnswer(text) {
    if (!text) {
        return "";
    }

    text = text.trim();

    text = text.replace(
        /^assistant\s*:\s*/i,
        ""
    );

    text = text.replace(
        /^luna\s*:\s*/i,
        ""
    );

    return text.trim();
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

    history.push({
        role: "user",
        content: userMessage
    });

    saveHistory();

    const memoryText =
        Object.keys(memory).length > 0
            ? JSON.stringify(memory)
            : "Nenhuma memória salva.";

    const recentHistory =
        history.slice(-8);

    const messages = [
        {
            role: "system",
            content:
                PERSONALITY +
                `

Memória atual:

${memoryText}

Responda somente ao usuário.
Não escreva instruções.
Não continue a conversa sozinho.
Não invente outra pessoa falando.
`
        },
        ...recentHistory
    ];

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
                    max_new_tokens: 100,
                    temperature: 0.7,
                    do_sample: true,
                    repetition_penalty: 1.1,
                    top_k: 20,
                    top_p: 0.8
                }
            );

        let answer = "";

        if (
            output &&
            output[0]
        ) {
            const generated =
                output[0].generated_text;

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
                    last &&
                    typeof last.content ===
                        "string"
                ) {
                    answer =
                        last.content;
                } else if (
                    typeof last ===
                        "string"
                ) {
                    answer =
                        last;
                }
            } else if (
                typeof generated ===
                    "string"
            ) {
                answer =
                    generated;
            }
        }

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

        let errorText =
            "Erro desconhecido";

        if (
            error &&
            error.message
        ) {
            errorText =
                error.message;
        }

        addMessage(
            "ERRO AO RESPONDER:\n\n" +
            errorText,
            "ai"
        );
    }

    generating = false;

    sendButton.disabled = false;

    input.focus();
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
    }
);

async function start() {
    loadHistory();

    await loadModel();
}

start();

import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct";

/*
 * Configuração conservadora para Chrome Android.
 *
 * Uma thread evita a multiplicação de buffers e o pico de RAM
 * que pode ocorrer com WASM multithread em aparelhos móveis.
 * O modelo é carregado somente uma vez e permanece em memória.
 */
env.backends.onnx.wasm.numThreads = 1;
/*
 * Nunca execute a inferência pesada na thread principal do navegador.
 * Com false, o Chrome Android pode parecer travado durante a segunda
 * geração, mesmo que o modelo ainda esteja funcionando.
 */
env.backends.onnx.wasm.proxy = true;

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
- Use palavras normais e frases naturais.
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

/* Limites deliberados para o hardware móvel. */
const MAX_GENERATED_TOKENS = 32;
const MAX_CONTEXT_CHARS = 3000;
const MAX_MEMORY_CHARS = 600;
const MAX_SAVED_HISTORY_ITEMS = 200;

let generator = null;
let modelLoading = false;
let generating = false;

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

let memory = readJson("luna_memory", {});
let history = readJson("luna_history", []);

if (!memory || typeof memory !== "object" || Array.isArray(memory)) {
    memory = {};
}

if (!Array.isArray(history)) {
    history = [];
}

function readJson(key, fallback) {
    try {
        const value = JSON.parse(
            localStorage.getItem(key) || JSON.stringify(fallback)
        );

        return value;
    } catch (error) {
        console.warn(`Dados locais inválidos em ${key}.`, error);
        return fallback;
    }
}

function setStatus(text) {
    status.textContent = text;
}

function saveMemory() {
    try {
        localStorage.setItem(
            "luna_memory",
            JSON.stringify(memory)
        );
    } catch (error) {
        console.warn("Não foi possível salvar a memória local.", error);
    }
}

function saveHistory() {
    /*
     * O histórico completo continua local, mas é limitado para evitar
     * que localStorage cresça indefinidamente e cause serializações
     * grandes a cada envio.
     */
    if (history.length > MAX_SAVED_HISTORY_ITEMS) {
        history = history.slice(-MAX_SAVED_HISTORY_ITEMS);
    }

    try {
        localStorage.setItem(
            "luna_history",
            JSON.stringify(history)
        );
    } catch (error) {
        console.warn("Não foi possível salvar o histórico local.", error);
    }
}

function addMessage(text, type) {
    const element = document.createElement("div");

    element.className = `message ${type}`;
    element.textContent = String(text || "");

    chat.appendChild(element);
    chat.scrollTop = chat.scrollHeight;

    return element;
}

function loadHistory() {
    for (const message of history) {
        if (!message || typeof message.content !== "string") {
            continue;
        }

        addMessage(
            message.content,
            message.role === "user" ? "user" : "ai"
        );
    }
}

function progressCallback(progress) {
    if (!progress) {
        return;
    }

    if (progress.status === "initiate") {
        setStatus("Iniciando download...");
        return;
    }

    if (progress.status === "progress") {
        const value = Number(progress.progress);

        setStatus(
            Number.isFinite(value)
                ? `Baixando IA... ${Math.round(value)}%`
                : "Baixando IA..."
        );

        return;
    }

    if (progress.status === "done") {
        setStatus("Finalizando...");
    }
}

async function loadModel() {
    if (generator || modelLoading) {
        return;
    }

    modelLoading = true;
    setStatus("Preparando IA...");

    try {
        setStatus("Iniciando Luna pela CPU...");

        generator = await pipeline(
            "text-generation",
            MODEL,
            {
                device: "wasm",
                dtype: "q8",
                progress_callback: progressCallback
            }
        );

        setStatus("Online - CPU WASM worker");
        addMessage("Luna está online.", "ai");
    } catch (error) {
        console.error("ERRO COMPLETO AO CARREGAR O MODELO:", error);

        generator = null;
        setStatus("Erro ao carregar IA");

        addMessage(
            `ERRO REAL:\n\n${getErrorText(error)}`,
            "ai"
        );
    } finally {
        modelLoading = false;
    }
}

function getErrorText(error) {
    if (error && error.message) {
        return error.message;
    }

    return String(error || "Erro desconhecido");
}

function getMemoryText() {
    let text;

    try {
        text = JSON.stringify(memory);
    } catch (error) {
        text = "{}";
    }

    if (!text || text === "{}") {
        return "Nenhuma memória salva.";
    }

    return text.slice(0, MAX_MEMORY_CHARS);
}

function isValidChatMessage(message) {
    return Boolean(
        message &&
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string" &&
        message.content.trim()
    );
}

function getPromptHistory() {
    /*
     * O histórico local não é apagado. Apenas uma janela curta é enviada
     * ao modelo para impedir que o prompt e o KV cache cresçam a cada turno.
     * O limite é por caracteres, uma aproximação segura para tokens.
     */
    const selected = [];
    let usedChars = 0;

    for (let index = history.length - 1; index >= 0; index -= 1) {
        const message = history[index];

        if (!isValidChatMessage(message)) {
            continue;
        }

        const content = message.content.trim();
        const cost = content.length + 40;

        if (
            selected.length > 0 &&
            usedChars + cost > MAX_CONTEXT_CHARS
        ) {
            break;
        }

        selected.push({
            role: message.role,
            content: content
        });

        usedChars += cost;
    }

    return selected.reverse();
}

function buildMessages() {
    return [
        {
            role: "system",
            content:
                `${PERSONALITY}\n\n` +
                `Memória da Luna:\n${getMemoryText()}\n\n` +
                "Responda apenas à última mensagem do usuário."
        },
        ...getPromptHistory()
    ];
}

function extractAnswer(output) {
    if (!output || !Array.isArray(output) || !output[0]) {
        return "";
    }

    const generated = output[0].generated_text;

    if (Array.isArray(generated)) {
        for (let index = generated.length - 1; index >= 0; index -= 1) {
            const message = generated[index];

            if (
                message &&
                message.role === "assistant" &&
                typeof message.content === "string"
            ) {
                return message.content.trim();
            }
        }

        return "";
    }

    return typeof generated === "string"
        ? generated.trim()
        : "";
}

function cleanAnswer(text) {
    if (!text) {
        return "";
    }

    return String(text)
        .replace(/^assistant\s*:\s*/i, "")
        .replace(/^luna\s*:\s*/i, "")
        .replace(/<\|im_end\|>[\s\S]*$/g, "")
        .replace(/<\|endoftext\|>[\s\S]*$/g, "")
        .trim();
}

async function generate(userMessage) {
    /* Guarda contra cliques rápidos e chamadas simultâneas. */
    if (!generator || generating || modelLoading) {
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

    const thinking = addMessage("Pensando...", "ai");
    let output = null;

    try {
        /*
         * O modelo é reutilizado. Ele não é carregado novamente aqui.
         * O chat template oficial do Qwen é aplicado pelo tokenizer.
         */
        output = await generator(
            buildMessages(),
            {
                max_new_tokens: MAX_GENERATED_TOKENS,
                do_sample: false,
                return_full_text: false,
                use_cache: false,
                eos_token_id: [151645, 151643],
                pad_token_id: 151643
            }
        );

        const answer = cleanAnswer(extractAnswer(output)) ||
            "Não consegui gerar uma resposta.";

        thinking.remove();
        addMessage(answer, "ai");

        history.push({
            role: "assistant",
            content: answer
        });

        saveHistory();
    } catch (error) {
        console.error("ERRO AO RESPONDER:", error);

        thinking.remove();
        addMessage(
            `ERRO AO RESPONDER:\n\n${getErrorText(error)}`,
            "ai"
        );
    } finally {
        /*
         * Não chamar dispose() aqui: isso destruiria os pesos do modelo
         * e obrigaria um novo carregamento. Apenas removemos a referência
         * ao resultado temporário para facilitar a coleta de lixo.
         */
        output = null;
        generating = false;
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

sendButton.addEventListener("click", async () => {
    const message = input.value.trim();

    if (!message || generating || modelLoading) {
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
    addMessage(message, "user");

    await generate(message);
});

input.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !generating
    ) {
        event.preventDefault();
        sendButton.click();
    }
});

clearButton.addEventListener("click", () => {
    if (generating) {
        return;
    }

    history = [];
    saveHistory();
    chat.innerHTML = "";

    setStatus(
        generator
            ? "Online - CPU WASM worker"
            : "Modelo não carregado"
    );
});

async function start() {
    loadHistory();
    await loadModel();
}

start();

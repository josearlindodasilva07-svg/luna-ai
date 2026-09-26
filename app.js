import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct";

/*
 * O proxy é essencial no Chrome Android: a inferência acontece em um
 * worker, e não na thread que desenha a interface.
 */
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = true;

const MODEL_DTYPE = "q4";
const MAX_NEW_TOKENS = 32;
const MAX_PROMPT_CHARS = 3000;
const MAX_MEMORY_CHARS = 600;
const MAX_STORED_MESSAGES = 200;

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
- Responda sempre em português brasileiro.
- Use palavras normais e frases naturais.
- Se a pergunta for simples, responda de forma simples.
- Não repita palavras ou frases sem necessidade.
- Não escreva palavras inventadas.
- Não misture idiomas.
- Não copie a mensagem do usuário.
- Não invente informações.
- Se não souber algo, diga que não sabe.
- Você não é uma pessoa humana.
`;

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

let generator = null;
let loading = false;
let generating = false;
let memory = loadJson("luna_memory", {});
let history = loadJson("luna_history", []);

if (!memory || typeof memory !== "object" || Array.isArray(memory)) {
    memory = {};
}

if (!Array.isArray(history)) {
    history = [];
}

function loadJson(key, fallback) {
    try {
        return JSON.parse(
            localStorage.getItem(key) || JSON.stringify(fallback)
        );
    } catch (error) {
        console.warn(`Falha ao ler ${key}.`, error);
        return fallback;
    }
}

function setStatus(text) {
    status.textContent = text;
}

function errorText(error) {
    return error && error.message
        ? error.message
        : String(error || "Erro desconhecido");
}

function saveHistory() {
    if (history.length > MAX_STORED_MESSAGES) {
        history = history.slice(-MAX_STORED_MESSAGES);
    }

    try {
        localStorage.setItem(
            "luna_history",
            JSON.stringify(history)
        );
    } catch (error) {
        console.warn("Falha ao salvar histórico local.", error);
    }
}

function saveMemory() {
    try {
        localStorage.setItem(
            "luna_memory",
            JSON.stringify(memory)
        );
    } catch (error) {
        console.warn("Falha ao salvar memória local.", error);
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
    for (const item of history) {
        if (!item || typeof item.content !== "string") {
            continue;
        }

        addMessage(
            item.content,
            item.role === "user" ? "user" : "ai"
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
    if (generator || loading) {
        return;
    }

    loading = true;
    setStatus("Preparando IA...");

    try {
        generator = await pipeline(
            "text-generation",
            MODEL,
            {
                device: "wasm",
                dtype: MODEL_DTYPE,
                progress_callback: progressCallback
            }
        );

        setStatus("Online - CPU WASM worker");
        addMessage("Luna está online.", "ai");
    } catch (error) {
        generator = null;
        setStatus("Erro ao carregar IA");
        console.error("ERRO AO CARREGAR MODELO:", error);
        addMessage(`ERRO REAL:\n\n${errorText(error)}`, "ai");
    } finally {
        loading = false;
    }
}

function memoryForPrompt() {
    let text = "{}";

    try {
        text = JSON.stringify(memory);
    } catch (error) {
        console.warn("Memória não serializável.", error);
    }

    if (!text || text === "{}") {
        return "Nenhuma memória salva.";
    }

    return text.slice(0, MAX_MEMORY_CHARS);
}

function validMessage(item) {
    return Boolean(
        item &&
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string" &&
        item.content.trim().length > 0
    );
}

function promptHistory() {
    const selected = [];
    let size = 0;

    for (let i = history.length - 1; i >= 0; i -= 1) {
        const item = history[i];

        if (!validMessage(item)) {
            continue;
        }

        const content = item.content.trim();
        const itemSize = content.length + 40;

        if (
            selected.length > 0 &&
            size + itemSize > MAX_PROMPT_CHARS
        ) {
            break;
        }

        selected.push({
            role: item.role,
            content
        });

        size += itemSize;
    }

    return selected.reverse();
}

function makeMessages() {
    return [
        {
            role: "system",
            content:
                `${PERSONALITY}\n\n` +
                `Memória da Luna:\n${memoryForPrompt()}\n\n` +
                "Responda somente à última mensagem do usuário."
        },
        ...promptHistory()
    ];
}

function extractAnswer(output) {
    if (!output || !Array.isArray(output) || !output[0]) {
        return "";
    }

    const generated = output[0].generated_text;

    if (Array.isArray(generated)) {
        for (let i = generated.length - 1; i >= 0; i -= 1) {
            const item = generated[i];

            if (
                item &&
                item.role === "assistant" &&
                typeof item.content === "string"
            ) {
                return item.content.trim();
            }
        }

        return "";
    }

    return typeof generated === "string"
        ? generated.trim()
        : "";
}

function cleanAnswer(text) {
    return String(text || "")
        .replace(/^assistant\s*:\s*/i, "")
        .replace(/^luna\s*:\s*/i, "")
        .replace(/<\|im_end\|>[\s\S]*$/g, "")
        .replace(/<\|endoftext\|>[\s\S]*$/g, "")
        .trim();
}

function setBusy(value) {
    generating = value;
    sendButton.disabled = value;
    input.disabled = value;
}

async function generate(userMessage) {
    if (!generator || loading || generating) {
        return;
    }

    setBusy(true);

    /*
     * A mensagem é inserida uma única vez no histórico, antes da chamada.
     * O elemento visual já foi criado pelo handler do botão.
     */
    history.push({
        role: "user",
        content: userMessage
    });
    saveHistory();

    const thinking = addMessage("Pensando...", "ai");
    const messages = makeMessages();

    try {
        /*
         * Importante: não usar use_cache:false.
         * Sem o KV cache, o modelo recalcula o prompt inteiro a cada token,
         * aumentando muito o uso de CPU e dando a impressão de congelamento.
         * O padrão use_cache:true mantém o cache somente durante esta chamada
         * e o pipeline o descarta ao terminar.
         */
        const output = await generator(
            messages,
            {
                max_new_tokens: MAX_NEW_TOKENS,
                do_sample: false,
                return_full_text: false,
                use_cache: true,
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
        console.error("ERRO DURANTE A GERAÇÃO:", error);
        thinking.remove();
        addMessage(
            `ERRO AO RESPONDER:\n\n${errorText(error)}`,
            "ai"
        );
    } finally {
        setBusy(false);
        input.focus();
    }
}

async function handleSend() {
    /*
     * O handler retorna antes de tocar no modelo se ainda estiver ocupado.
     * Isso evita duas chamadas de generator no mesmo pipeline.
     */
    if (loading || generating || !generator) {
        if (!generator && !loading) {
            addMessage(
                "A Luna ainda não terminou de carregar.",
                "ai"
            );
        }
        return;
    }

    const message = input.value.trim();

    if (!message) {
        return;
    }

    input.value = "";
    addMessage(message, "user");

    /* Deixa o navegador pintar a mensagem antes do trabalho do generator. */
    await new Promise(resolve => setTimeout(resolve, 0));
    await generate(message);
}

sendButton.addEventListener("click", handleSend);

input.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {
        event.preventDefault();

        if (!generating && !loading) {
            handleSend();
        }
    }
});

clearButton.addEventListener("click", () => {
    if (generating || loading) {
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

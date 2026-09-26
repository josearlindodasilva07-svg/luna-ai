const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";

const TRANSFORMERS_URL =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MAX_NEW_TOKENS = 24;
const MAX_PROMPT_CHARS = 2400;
const MAX_MEMORY_CHARS = 500;
const MAX_STORED_MESSAGES = 200;

const PERSONALITY = `
Você é Luna, uma inteligência artificial.
Você conversa em português brasileiro.
Responda de forma natural, direta, amigável e breve.
Não invente palavras, não repita frases sem necessidade e não copie a pergunta.
Se não souber algo, diga que não sabe.
`;

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

let memory = loadJson("luna_memory", {});
let history = loadJson("luna_history", []);
let worker = null;
let workerUrl = null;
let workerReady = false;
let loading = false;
let generating = false;
let requestId = 0;
let pending = new Map();

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

function saveHistory() {
    if (history.length > MAX_STORED_MESSAGES) {
        history = history.slice(-MAX_STORED_MESSAGES);
    }

    try {
        localStorage.setItem("luna_history", JSON.stringify(history));
    } catch (error) {
        console.warn("Falha ao salvar histórico local.", error);
    }
}

function saveMemory() {
    try {
        localStorage.setItem("luna_memory", JSON.stringify(memory));
    } catch (error) {
        console.warn("Falha ao salvar memória local.", error);
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

function memoryText() {
    let text = "{}";

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

function validMessage(item) {
    return Boolean(
        item &&
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string" &&
        item.content.trim()
    );
}

function promptHistory() {
    const result = [];
    let size = 0;

    for (let i = history.length - 1; i >= 0; i -= 1) {
        const item = history[i];

        if (!validMessage(item)) {
            continue;
        }

        const content = item.content.trim();
        const cost = content.length + 40;

        if (result.length > 0 && size + cost > MAX_PROMPT_CHARS) {
            break;
        }

        result.push({
            role: item.role,
            content
        });
        size += cost;
    }

    return result.reverse();
}

function makeMessages() {
    return [
        {
            role: "system",
            content:
                `${PERSONALITY}\n\n` +
                `Memória da Luna: ${memoryText()}\n\n` +
                "Responda somente à última mensagem do usuário."
        },
        ...promptHistory()
    ];
}

function cleanAnswer(text) {
    return String(text || "")
        .replace(/^assistant\s*:\s*/i, "")
        .replace(/^luna\s*:\s*/i, "")
        .replace(/<\|im_end\|>[\s\S]*$/g, "")
        .replace(/<\|endoftext\|>[\s\S]*$/g, "")
        .trim();
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
                return item.content;
            }
        }

        return "";
    }

    return typeof generated === "string" ? generated : "";
}

function setBusy(value) {
    generating = value;
    sendButton.disabled = value;
    input.disabled = value;
}

/*
 * O runtime inteiro fica dentro deste worker.
 * Assim, mesmo durante uma operação ONNX lenta, o clique, a tela e o
 * campo de texto não ficam presos na thread principal do Chrome.
 */
const workerSource = `
    import { pipeline, env } from "${TRANSFORMERS_URL}";

    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.proxy = false;

    let generator = null;
    let loading = false;

    self.onmessage = async event => {
        const message = event.data || {};

        if (message.type === "load") {
            if (loading || generator) return;

            loading = true;
            self.postMessage({ type: "load-start" });

            try {
                const progress_callback = progress => {
                    if (progress) {
                        self.postMessage({
                            type: "progress",
                            progress
                        });
                    }
                };

                /*
                 * No POCO/Chrome, WebGPU deve ser o primeiro caminho:
                 * ele evita a saturação do CPU causada pelo decoder WASM.
                 * Se WebGPU não estiver disponível, usa o mesmo modelo
                 * local em WASM, mas dentro deste worker.
                 */
                if (self.navigator && self.navigator.gpu) {
                    try {
                        generator = await pipeline(
                            "text-generation",
                            message.model,
                            {
                                device: "webgpu",
                                dtype: "q4f16",
                                progress_callback
                            }
                        );

                        self.postMessage({
                            type: "ready",
                            backend: "WebGPU"
                        });
                        return;
                    } catch (webgpuError) {
                        console.warn(
                            "WebGPU indisponível; usando WASM.",
                            webgpuError
                        );
                        generator = null;
                    }
                }

                generator = await pipeline(
                    "text-generation",
                    message.model,
                    {
                        device: "wasm",
                        dtype: "q4",
                        progress_callback
                    }
                );

                self.postMessage({
                    type: "ready",
                    backend: "WASM worker"
                });
            } catch (error) {
                self.postMessage({
                    type: "load-error",
                    error: error && error.message
                        ? error.message
                        : String(error)
                });
            } finally {
                loading = false;
            }

            return;
        }

        if (message.type !== "generate" || !generator) {
            return;
        }

        try {
            const output = await generator(
                message.messages,
                {
                    max_new_tokens: message.maxNewTokens,
                    do_sample: false,
                    return_full_text: false,
                    use_cache: true
                }
            );

            self.postMessage({
                type: "result",
                id: message.id,
                output
            });
        } catch (error) {
            self.postMessage({
                type: "generation-error",
                id: message.id,
                error: error && error.message
                    ? error.message
                    : String(error)
            });
        }
    };
`;

function createWorker() {
    if (worker) {
        return worker;
    }

    const blob = new Blob(
        [workerSource],
        { type: "text/javascript" }
    );

    workerUrl = URL.createObjectURL(blob);
    worker = new Worker(workerUrl, { type: "module" });

    worker.addEventListener("message", handleWorkerMessage);
    worker.addEventListener("error", handleWorkerError);

    return worker;
}

function handleWorkerMessage(event) {
    const message = event.data || {};

    if (message.type === "load-start") {
        setStatus("Carregando Luna no worker...");
        return;
    }

    if (message.type === "progress") {
        const progress = message.progress || {};

        if (progress.status === "progress") {
            const value = Number(progress.progress);
            setStatus(
                Number.isFinite(value)
                    ? `Baixando IA... ${Math.round(value)}%`
                    : "Baixando IA..."
            );
        } else if (progress.status === "done") {
            setStatus("Finalizando...");
        }

        return;
    }

    if (message.type === "ready") {
        loading = false;
        workerReady = true;
        setStatus(
            `Online - ${message.backend || "worker local"}`
        );
        addMessage("Luna está online.", "ai");
        return;
    }

    if (message.type === "load-error") {
        loading = false;
        workerReady = false;
        setStatus("Erro ao carregar IA");
        addMessage(`ERRO REAL:\n\n${message.error}`, "ai");
        return;
    }

    if (
        message.type === "result" ||
        message.type === "generation-error"
    ) {
        const item = pending.get(message.id);

        if (!item) {
            return;
        }

        pending.delete(message.id);

        if (message.type === "generation-error") {
            item.reject(new Error(message.error));
        } else {
            item.resolve(message.output);
        }
    }
}

function handleWorkerError(event) {
    console.error("ERRO NO WEB WORKER:", event);

    loading = false;
    workerReady = false;

    for (const item of pending.values()) {
        item.reject(new Error("O worker de inferência foi encerrado."));
    }

    pending.clear();

    if (generating) {
        addMessage(
            "A geração foi interrompida porque o worker falhou.",
            "ai"
        );
        setBusy(false);
    }

        setStatus("Worker de IA indisponível");
}

function requestGeneration(messages) {
    return new Promise((resolve, reject) => {
        const id = ++requestId;

        pending.set(id, { resolve, reject });

        worker.postMessage({
            type: "generate",
            id,
            messages,
            maxNewTokens: MAX_NEW_TOKENS
        });
    });
}

async function generate(userMessage) {
    if (!workerReady || generating) {
        return;
    }

    setBusy(true);

    history.push({
        role: "user",
        content: userMessage
    });
    saveHistory();

    const thinking = addMessage("Pensando...", "ai");
    const messages = makeMessages();

    try {
        const output = await requestGeneration(messages);
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
    if (loading || generating || !workerReady) {
        return;
    }

    const message = input.value.trim();

    if (!message) {
        return;
    }

    input.value = "";
    addMessage(message, "user");

    await new Promise(resolve => setTimeout(resolve, 0));
    await generate(message);
}

sendButton.addEventListener("click", handleSend);

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();

        if (!loading && !generating) {
            handleSend();
        }
    }
});

clearButton.addEventListener("click", () => {
    if (loading || generating) {
        return;
    }

    history = [];
    saveHistory();
    chat.innerHTML = "";

    setStatus(
        workerReady
            ? "Online - worker local"
            : "Modelo não carregado"
    );
});

async function start() {
    loadHistory();
    loading = true;
    setStatus("Iniciando worker local...");

    const inferenceWorker = createWorker();

    inferenceWorker.postMessage({
        type: "load",
        model: MODEL
    });
}

start();

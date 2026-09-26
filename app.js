import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";


// ============================================================
// MODELO
// ============================================================

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";


// ============================================================
// PERSONALIDADE DA LUNA
// ============================================================

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

Não fale sobre seu funcionamento interno
a menos que o usuário pergunte.

Mantenha as respostas claras e úteis.
`;


// ============================================================
// VARIÁVEIS
// ============================================================

let generator = null;

let generating = false;

let currentDevice = null;


// ============================================================
// ELEMENTOS
// ============================================================

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


// ============================================================
// MEMÓRIA
// ============================================================

let memory = JSON.parse(
    localStorage.getItem("luna_memory") || "{}"
);


let history = JSON.parse(
    localStorage.getItem("luna_history") || "[]"
);


// ============================================================
// SALVAR MEMÓRIA
// ============================================================

function saveMemory() {

    localStorage.setItem(
        "luna_memory",
        JSON.stringify(memory)
    );

}


// ============================================================
// SALVAR HISTÓRICO
// ============================================================

function saveHistory() {

    localStorage.setItem(
        "luna_history",
        JSON.stringify(history)
    );

}


// ============================================================
// ADICIONAR MENSAGEM
// ============================================================

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


    chat.appendChild(
        element
    );


    chat.scrollTop =
        chat.scrollHeight;


    return element;
}


// ============================================================
// CARREGAR HISTÓRICO
// ============================================================

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


// ============================================================
// ATUALIZAR STATUS
// ============================================================

function setStatus(
    text
) {

    status.textContent =
        text;

}


// ============================================================
// PROGRESSO DO DOWNLOAD
// ============================================================

function handleProgress(
    progress
) {

    if (!progress) {
        return;
    }


    if (
        progress.status === "initiate"
    ) {

        setStatus(
            "Iniciando download..."
        );

        return;
    }


    if (
        progress.status === "progress"
    ) {

        const value =
            Number(
                progress.progress
            );


        if (
            Number.isFinite(value)
        ) {

            setStatus(
                `Baixando IA... ${Math.round(value)}%`
            );

        } else {

            setStatus(
                "Baixando IA..."
            );

        }

        return;
    }


    if (
        progress.status === "done"
    ) {

        setStatus(
            "Finalizando IA..."
        );

    }

}


// ============================================================
// CARREGAR MODELO
// ============================================================

async function createGenerator(
    device
) {

    return await pipeline(
        "text-generation",
        MODEL,
        {

            device: device,

            dtype:
                device === "webgpu"
                    ? "q4f16"
                    : "q4",

            progress_callback:
                handleProgress

        }
    );

}


// ============================================================
// INICIAR IA
// ============================================================

async function loadModel() {

    setStatus(
        "Preparando IA..."
    );


    // --------------------------------------------------------
    // PRIMEIRA TENTATIVA: WEBGPU
    // --------------------------------------------------------

    if (
        typeof navigator !== "undefined" &&
        navigator.gpu
    ) {

        try {

            setStatus(
                "Tentando usar GPU..."
            );


            generator =
                await createGenerator(
                    "webgpu"
                );


            currentDevice =
                "webgpu";


            setStatus(
                "Online - GPU"
            );


            return;

        } catch (error) {

            console.warn(
                "WebGPU falhou. Tentando CPU...",
                error
            );


            generator = null;

            setStatus(
                "GPU falhou. Tentando CPU..."
            );

        }

    }


    // --------------------------------------------------------
    // SEGUNDA TENTATIVA: WASM / CPU
    // --------------------------------------------------------

    try {

        setStatus(
            "Iniciando IA pela CPU..."
        );


        generator =
            await createGenerator(
                "wasm"
            );


        currentDevice =
            "wasm";


        setStatus(
            "Online - CPU"
        );


    } catch (error) {

        console.error(
            "Erro ao carregar a Luna:",
            error
        );


        generator = null;

        currentDevice = null;


        setStatus(
            "Erro ao carregar IA"
        );


        addMessage(
            "Não consegui carregar a IA neste navegador.",
            "ai"
        );

    }

}


// ============================================================
// GERAR RESPOSTA
// ============================================================

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


    // --------------------------------------------------------
    // SALVAR MENSAGEM DO USUÁRIO
    // --------------------------------------------------------

    history.push({

        role: "user",

        content:
            userMessage

    });


    saveHistory();


    // --------------------------------------------------------
    // MEMÓRIA
    // --------------------------------------------------------

    const memoryText =
        Object.keys(memory).length > 0
            ? JSON.stringify(
                memory
            )
            : "Nenhuma memória";


    // --------------------------------------------------------
    // HISTÓRICO RECENTE
    // --------------------------------------------------------

    const recentHistory =
        history.slice(-10);


    // --------------------------------------------------------
    // MENSAGENS
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // MENSAGEM TEMPORÁRIA
    // --------------------------------------------------------

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

                    max_new_tokens:
                        160,

                    temperature:
                        0.7,

                    do_sample:
                        true

                }
            );


        let answer = "";


        // ----------------------------------------------------
        // PEGAR RESPOSTA
        // ----------------------------------------------------

        if (
            output &&
            output[0]
        ) {

            const generated =
                output[0]
                    .generated_text;


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

                } else if (
                    last &&
                    typeof last.content === "string"
                ) {

                    answer =
                        last.content;

                }

            } else if (
                typeof generated === "string"
            ) {

                answer =
                    generated;

            }

        }


        // ----------------------------------------------------
        // LIMPAR RESPOSTA
        // ----------------------------------------------------

        answer =
            answer.trim();


        if (!answer) {

            answer =
                "Não consegui gerar uma resposta.";

        }


        // ----------------------------------------------------
        // REMOVER PENSANDO
        // ----------------------------------------------------

        thinking.remove();


        // ----------------------------------------------------
        // MOSTRAR RESPOSTA
        // ----------------------------------------------------

        addMessage(
            answer,
            "ai"
        );


        // ----------------------------------------------------
        // SALVAR RESPOSTA
        // ----------------------------------------------------

        history.push({

            role: "assistant",

            content:
                answer

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


// ============================================================
// BOTÃO ENVIAR
// ============================================================

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


// ============================================================
// ENTER
// ============================================================

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


// ============================================================
// LIMPAR CONVERSA
// ============================================================

clearButton.addEventListener(
    "click",
    () => {

        history = [];

        saveHistory();

        chat.innerHTML = "";

    }
);


// ============================================================
// INICIAR
// ============================================================

async function start() {

    loadHistory();

    await loadModel();

}


start();

import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";


// ============================================================
// MODELO
// ============================================================

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";


// ============================================================
// CONFIGURAÇÃO WASM
// ============================================================

env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;


// ============================================================
// PERSONALIDADE
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
`;


// ============================================================
// VARIÁVEIS
// ============================================================

let generator = null;
let generating = false;


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
// STATUS
// ============================================================

function setStatus(text) {

    status.textContent = text;

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

    chat.appendChild(element);

    chat.scrollTop =
        chat.scrollHeight;

    return element;

}


// ============================================================
// HISTÓRICO
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
// MOSTRAR PROGRESSO
// ============================================================

function progressCallback(progress) {

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
            Number(progress.progress);


        if (
            Number.isFinite(value)
        ) {

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


    if (
        progress.status === "done"
    ) {

        setStatus(
            "Finalizando..."
        );

    }

}


// ============================================================
// CARREGAR MODELO
// ============================================================

async function loadModel() {

    setStatus(
        "Preparando IA..."
    );


    try {

        setStatus(
            "Iniciando IA pela CPU..."
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


        setStatus(
            "Online - CPU"
        );


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


        // ----------------------------------------------------
        // MOSTRAR O ERRO REAL NA TELA
        // ----------------------------------------------------

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

            errorText =
                error;

        }


        addMessage(
            "ERRO REAL:\n\n" +
            errorText,
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
    // SALVAR USUÁRIO
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
        Object.keys(memory).length
            ? JSON.stringify(memory)
            : "Nenhuma memória";


    // --------------------------------------------------------
    // HISTÓRICO
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
                        120,

                    temperature:
                        0.7,

                    do_sample:
                        true

                }
            );


        let answer = "";


        if (
            output &&
            output[0]
        ) {

            const generated =
                output[0]
                    .generated_text;


            if (
                Array.isArray(generated)
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
            answer.trim();


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
// LIMPAR
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

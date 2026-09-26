import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";


const MODEL =
    "HuggingFaceTB/SmolLM2-360M-Instruct";


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


/* =========================
   HISTÓRICO
========================= */

let history = JSON.parse(
    localStorage.getItem("luna_history") || "[]"
);


/* =========================
   SALVAR MEMÓRIA
========================= */

function saveMemory() {

    localStorage.setItem(
        "luna_memory",
        JSON.stringify(memory)
    );
}


/* =========================
   SALVAR HISTÓRICO
========================= */

function saveHistory() {

    localStorage.setItem(
        "luna_history",
        JSON.stringify(history)
    );
}


/* =========================
   MOSTRAR MENSAGEM
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
   CARREGAR IA
========================= */

async function loadModel() {

    status.textContent =
        "Baixando IA...";

    generator =
        await pipeline(
            "text-generation",
            MODEL,
            {
                device: "webgpu"
            }
        );

    status.textContent =
        "Online";
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
       SALVAR USUÁRIO
    */

    history.push({

        role: "user",

        content: userMessage
    });


    saveHistory();


    /*
       MEMÓRIA
    */

    const memoryText =
        Object.keys(memory).length
            ? JSON.stringify(memory)
            : "Nenhuma memória";


    /*
       PEGAR ÚLTIMAS MENSAGENS
    */

    const recentHistory =
        history.slice(-12);


    /*
       CONTEXTO
    */

    const messages = [

        {
            role: "system",

            content:
                PERSONALITY +

                `

Memória:

${memoryText}
`
        },

        ...recentHistory
    ];


    /*
       GERAR
    */

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
                    typeof last === "string"
                ) {

                    answer = last;

                } else {

                    answer =
                        last.content || "";
                }
            }
        }


        if (!answer) {

            answer =
                "Não consegui gerar uma resposta.";
        }


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

        console.error(error);

        addMessage(
            "Erro ao gerar resposta.",
            "ai"
        );
    }


    generating = false;

    sendButton.disabled = false;
}


/* =========================
   ENVIAR
========================= */

sendButton.addEventListener(
    "click",
    async () => {

        const message =
            input.value.trim();


        if (!message) {
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
   ENTER
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
   LIMPAR CHAT
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
   INICIAR
========================= */

async function start() {

    loadHistory();

    try {

        await loadModel();

    } catch (error) {

        console.error(error);

        status.textContent =
            "WebGPU não disponível";

        addMessage(
            "Seu navegador não conseguiu iniciar a IA pela GPU.",
            "ai"
        );
    }
}


start();

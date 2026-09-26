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


let modelReady = false;
let generating = false;


/* ============================================================
   WORKER
   ============================================================ */

const worker =
    new Worker(
        "./worker.js?v=22",
        {
            type: "module"
        }
    );


/* ============================================================
   MENSAGENS
   ============================================================ */

function addMessage(text, type) {

    const div =
        document.createElement("div");

    div.className =
        "message " + type;

    div.textContent =
        text;

    chat.appendChild(div);

    chat.scrollTop =
        chat.scrollHeight;
}


/* ============================================================
   STATUS
   ============================================================ */

function setStatus(text) {

    status.textContent =
        text;
}


/* ============================================================
   ERRO DO WORKER
   ============================================================ */

worker.onerror =
    (event) => {

        modelReady = false;
        generating = false;

        sendButton.disabled = false;
        input.disabled = false;

        setStatus("Erro no Worker");

        addMessage(
            "ERRO NO WORKER: " +
            (
                event.message ||
                "Não foi possível iniciar o motor da Luna."
            ),
            "ai"
        );

        console.error(
            "Luna Worker Error:",
            event
        );
    };


/* ============================================================
   MENSAGENS DO WORKER
   ============================================================ */

worker.onmessage =
    (event) => {

        const data =
            event.data;


        if (
            data.type === "status"
        ) {

            setStatus(
                data.text
            );

            return;
        }


        if (
            data.type === "loaded"
        ) {

            modelReady = true;
            generating = false;

            setStatus(
                "Online"
            );

            sendButton.disabled = false;
            input.disabled = false;

            return;
        }


        if (
            data.type === "result"
        ) {

            generating = false;

            sendButton.disabled = false;
            input.disabled = false;

            setStatus(
                "Online"
            );


            const answer =
                String(
                    data.text || ""
                ).trim();


            addMessage(
                answer ||
                "Não consegui responder.",
                "ai"
            );

            return;
        }


        if (
            data.type === "error"
        ) {

            generating = false;

            sendButton.disabled = false;
            input.disabled = false;

            setStatus(
                "Erro"
            );

            addMessage(
                "ERRO AO RESPONDER: " +
                data.error,
                "ai"
            );

            console.error(
                "Luna:",
                data.error
            );

            return;
        }
    };


/* ============================================================
   ENVIAR
   ============================================================ */

function sendMessage() {

    if (generating) {
        return;
    }


    if (!modelReady) {

        addMessage(
            "O modelo ainda está carregando.",
            "ai"
        );

        return;
    }


    const text =
        input.value.trim();


    if (!text) {
        return;
    }


    addMessage(
        text,
        "user"
    );


    input.value = "";

    generating = true;

    sendButton.disabled = true;
    input.disabled = true;

    setStatus(
        "Pensando..."
    );


    worker.postMessage({

        type: "generate",

        prompt: text

    });
}


/* ============================================================
   BOTÃO ENVIAR
   ============================================================ */

sendButton.addEventListener(
    "click",
    sendMessage
);


/* ============================================================
   ENTER
   ============================================================ */

input.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


/* ============================================================
   LIMPAR
   ============================================================ */

clearButton.addEventListener(
    "click",
    () => {

        chat.innerHTML = "";

    }
);


/* ============================================================
   INÍCIO
   ============================================================ */

sendButton.disabled = true;
input.disabled = true;

setStatus(
    "Carregando Luna..."
);


worker.postMessage({
    type: "load"
});

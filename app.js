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
        "./worker.js?v=25",
        {
            type: "module"
        }
    );


/* ============================================================
   MENSAGEM
   ============================================================ */

function addMessage(
    text,
    type
) {

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

function setStatus(
    text
) {

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

        sendButton.disabled = true;
        input.disabled = true;

        setStatus(
            "Erro no Worker"
        );

        addMessage(
            "ERRO NO WORKER:\n" +
            (
                event.message ||
                "Não foi possível iniciar o Worker."
            ),
            "ai"
        );

        console.error(
            "Worker error:",
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

            sendButton.disabled =
                false;

            input.disabled =
                false;

            return;
        }


        if (
            data.type === "result"
        ) {

            generating = false;

            setStatus(
                "Online"
            );

            sendButton.disabled =
                false;

            input.disabled =
                false;


            addMessage(
                data.text ||
                "Não consegui responder.",
                "ai"
            );

            return;
        }


        if (
            data.type === "error"
        ) {

            modelReady = false;
            generating = false;

            sendButton.disabled =
                true;

            input.disabled =
                true;

            setStatus(
                "Erro"
            );


            addMessage(
                "ERRO AO CARREGAR O MODELO:\n\n" +
                data.error,
                "ai"
            );


            console.error(
                "Luna error:",
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
            "A Luna ainda está carregando.",
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


    input.value =
        "";


    generating =
        true;


    sendButton.disabled =
        true;

    input.disabled =
        true;


    setStatus(
        "Pensando..."
    );


    worker.postMessage({

        type:
            "generate",

        prompt:
            text

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

        chat.innerHTML =
            "";

    }
);


/* ============================================================
   INÍCIO
   ============================================================ */

modelReady = false;
generating = false;

sendButton.disabled = true;
input.disabled = true;

setStatus(
    "Iniciando..."
);


worker.postMessage({
    type: "load"
});

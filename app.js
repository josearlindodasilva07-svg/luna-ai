const MODEL =
"onnx-community/LFM2.5-350M-ONNX";

const TRANSFORMERS_URL =
"https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

const workerCode = `
import {
pipeline,
env
} from "${TRANSFORMERS_URL}";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let generator = null;
let loading = false;

self.onmessage = async (event) => {

const data = event.data;

try {

    if (data.type === "load") {

        if (generator) {

            self.postMessage({
                type: "loaded"
            });

            return;
        }

        if (loading) {
            return;
        }

        loading = true;


        self.postMessage({
            type: "status",
            text: "Carregando Luna..."
        });


        generator = await pipeline(
            "text-generation",
            "${MODEL}",
            {
                device: "wasm",
                dtype: "q4"
            }
        );


        loading = false;


        self.postMessage({
            type: "loaded"
        });

        return;
    }


    if (data.type === "generate") {

        if (!generator) {

            throw new Error(
                "Modelo ainda não foi carregado."
            );
        }


        const userText =
            String(
                data.prompt || ""
            ).trim();


        if (!userText) {
            return;
        }


        const messages = [

            {
                role: "system",
                content:
                    "Você é Luna, uma assistente de IA. " +
                    "Responda em português do Brasil. " +
                    "Seja natural, simples, direta e útil. " +
                    "Responda somente ao que o usuário perguntar. " +
                    "Não invente informações."
            },

            {
                role: "user",
                content: userText
            }

        ];


        self.postMessage({
            type: "status",
            text: "Pensando..."
        });


        const result =
            await generator(
                messages,
                {
                    max_new_tokens: 100,

                    do_sample: true,

                    temperature: 0.1,

                    top_k: 50,

                    repetition_penalty: 1.05
                }
            );


        let text = "";


        if (
            Array.isArray(result) &&
            result.length > 0
        ) {

            const generated =
                result[0]?.generated_text;


            if (
                Array.isArray(generated) &&
                generated.length > 0
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

                    text =
                        last.content;
                }

            } else if (
                typeof generated ===
                "string"
            ) {

                text =
                    generated;
            }
        }


        text =
            String(text).trim();


        text =
            text
                .replace(
                    /^assistant\\s*:/i,
                    ""
                )
                .replace(
                    /^luna\\s*:/i,
                    ""
                )
                .replace(
                    /<\\|im_end\\|>/gi,
                    ""
                )
                .replace(
                    /<\\|endoftext\\|>/gi,
                    ""
                )
                .trim();


        if (!text) {

            text =
                "Não consegui responder.";
        }


        self.postMessage({
            type: "result",
            text
        });

    }

} catch (error) {

    loading = false;

    self.postMessage({
        type: "error",

        error:
            error?.message ||
            String(error),

        stack:
            error?.stack ||
            ""
    });
}

};

const workerBlob =
new Blob(
[workerCode],
{
type: "application/javascript"
}
);

const worker =
new Worker(
URL.createObjectURL(
workerBlob
),
{
type: "module"
}
);

let modelReady = false;
let generating = false;

function addMessage(
text,
type
) {

const div =
    document.createElement(
        "div"
    );

div.className =
    "message " + type;

div.textContent =
    text;

chat.appendChild(div);

chat.scrollTop =
    chat.scrollHeight;

return div;

}

function setStatus(text) {

status.textContent =
    text;

}

worker.onmessage =
(event) => {

    const data =
        event.data;


    if (
        data.type ===
        "status"
    ) {

        setStatus(
            data.text
        );

        return;
    }


    if (
        data.type ===
        "loaded"
    ) {

        modelReady =
            true;

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
        data.type ===
        "result"
    ) {

        generating =
            false;

        sendButton.disabled =
            false;

        input.disabled =
            false;

        setStatus(
            "Online"
        );


        let answer =
            String(
                data.text || ""
            ).trim();


        if (!answer) {

            answer =
                "Não consegui responder.";
        }


        addMessage(
            answer,
            "ai"
        );

        return;
    }


    if (
        data.type ===
        "error"
    ) {

        generating =
            false;

        sendButton.disabled =
            false;

        input.disabled =
            false;

        setStatus(
            "Erro"
        );


        addMessage(
            "ERRO AO RESPONDER: " +
            data.error,
            "ai"
        );


        console.error(
            "Luna Worker Error:",
            data.error
        );


        console.error(
            data.stack || ""
        );

        return;
    }
};

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
    type: "generate",
    prompt: text
});

}

sendButton.addEventListener(
"click",
sendMessage
);

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

clearButton.addEventListener(
"click",
() => {

    chat.innerHTML = "";
}

);

sendButton.disabled =
true;

input.disabled =
true;

setStatus(
"Carregando Luna..."
);

worker.postMessage({
type: "load"
});

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";

const TRANSFORMERS_URL =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const chat = document.getElementById("chat");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const status = document.getElementById("status");

const workerCode = `
import {
    AutoTokenizer,
    AutoModelForCausalLM,
    env
} from "${TRANSFORMERS_URL}";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let tokenizer = null;
let model = null;
let loading = false;

self.onmessage = async (event) => {

    const data = event.data;

    try {

        if (data.type === "load") {

            if (model && tokenizer) {

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

            tokenizer =
                await AutoTokenizer.from_pretrained(
                    "${MODEL}"
                );

            self.postMessage({
                type: "status",
                text: "Carregando modelo..."
            });

            model =
                await AutoModelForCausalLM.from_pretrained(
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

            if (!model || !tokenizer) {

                throw new Error(
                    "Modelo ainda não foi carregado."
                );
            }

            const userText =
                String(data.prompt || "").trim();

            if (!userText) {
                return;
            }


            /*
             * Prompt extremamente simples.
             *
             * O modelo é pequeno, então evitamos
             * instruções enormes.
             */

            const messages = [
                {
                    role: "system",
                    content:
                        "Você é Luna. " +
                        "Você fala somente português do Brasil. " +
                        "Você responde diretamente ao usuário. " +
                        "Nunca fale sobre ser uma guia do universo. " +
                        "Nunca diga que é uma entidade ou personagem. " +
                        "Nunca comece uma resposta com 'I am Luna'. " +
                        "Nunca responda em inglês."
                },
                {
                    role: "user",
                    content: userText
                }
            ];


            let prompt;

            if (
                typeof tokenizer.apply_chat_template ===
                "function"
            ) {

                prompt =
                    tokenizer.apply_chat_template(
                        messages,
                        {
                            tokenize: false,
                            add_generation_prompt: true
                        }
                    );

            } else {

                prompt =
                    "<|im_start|>system\\n" +
                    "Você é Luna. " +
                    "Fale somente português do Brasil. " +
                    "Responda diretamente ao usuário. " +
                    "Nunca responda em inglês." +
                    "<|im_end|>\\n" +

                    "<|im_start|>user\\n" +
                    userText +
                    "<|im_end|>\\n" +

                    "<|im_start|>assistant\\n";
            }


            const inputs =
                await tokenizer(
                    prompt,
                    {
                        return_tensors: "pt"
                    }
                );


            const output =
                await model.generate({
                    ...inputs,

                    max_new_tokens: 60,

                    do_sample: true,

                    temperature: 0.45,

                    top_p: 0.85,

                    repetition_penalty: 1.15
                });


            let text =
                tokenizer.decode(
                    output[0],
                    {
                        skip_special_tokens: true
                    }
                );


            /*
             * Remove o prompt caso ele tenha
             * sido devolvido junto com a resposta.
             */

            if (text.startsWith(prompt)) {

                text =
                    text.substring(
                        prompt.length
                    );
            }


            /*
             * Limpeza dos marcadores.
             */

            text = text
                .replace(
                    /<\\|im_start\\|>/gi,
                    ""
                )
                .replace(
                    /<\\|im_end\\|>/gi,
                    ""
                )
                .replace(
                    /<\\|assistant\\|>/gi,
                    ""
                )
                .replace(
                    /<\\|user\\|>/gi,
                    ""
                )
                .replace(
                    /<\\|system\\|>/gi,
                    ""
                )
                .trim();


            /*
             * Se o modelo começar a repetir
             * a personalidade em inglês,
             * corta essa resposta.
             */

            const badStarts = [
                "I am Luna",
                "I am a wise",
                "I am the one",
                "I have been listening",
                "I have always",
                "I will not stray",
                "The stories are",
                "I am here to guide",
                "I am your guide"
            ];


            for (const bad of badStarts) {

                if (
                    text
                        .toLowerCase()
                        .startsWith(
                            bad.toLowerCase()
                        )
                ) {

                    text = "";
                    break;
                }
            }


            /*
             * Evita respostas absurdamente longas
             * ou repetitivas.
             */

            if (text.length > 600) {

                text =
                    text.substring(
                        0,
                        600
                    ).trim();
            }


            /*
             * Remove repetições exatas de frases.
             */

            const sentences =
                text
                    .split(/(?<=[.!?])\\s+/)
                    .filter(Boolean);

            const unique = [];

            for (const sentence of sentences) {

                if (
                    !unique.some(
                        x =>
                            x.toLowerCase() ===
                            sentence.toLowerCase()
                    )
                ) {

                    unique.push(sentence);
                }
            }

            text =
                unique.join(" ").trim();


            if (!text) {

                text =
                    "não entendi direito, fala de novo";
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
                    "não consegui responder";
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

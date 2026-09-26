import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";


env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;


const MODEL =
    "onnx-community/LFM2.5-350M-ONNX";


let generator = null;
let loading = false;


/* ============================================================
   MENSAGENS
   ============================================================ */

self.onmessage =
    async (event) => {

        const data =
            event.data;


        try {

            /* ==================================================
               CARREGAR MODELO
               ================================================== */

            if (
                data.type === "load"
            ) {

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
                    text: "Baixando modelo..."
                });


                generator =
                    await pipeline(
                        "text-generation",
                        MODEL,
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


            /* ==================================================
               GERAR RESPOSTA
               ================================================== */

            if (
                data.type === "generate"
            ) {

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


                self.postMessage({
                    type: "status",
                    text: "Pensando..."
                });


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

                        content:
                            userText
                    }

                ];


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
                            /^assistant\s*:/i,
                            ""
                        )
                        .replace(
                            /^luna\s*:/i,
                            ""
                        )
                        .replace(
                            /<\|im_end\|>/gi,
                            ""
                        )
                        .replace(
                            /<\|endoftext\|>/gi,
                            ""
                        )
                        .trim();


                if (!text) {

                    text =
                        "Não consegui responder.";
                }


                self.postMessage({

                    type: "result",

                    text: text

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

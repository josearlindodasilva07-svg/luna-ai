import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

self.postMessage({
    type: "status",
    text: "Worker OK"
});

try {

    self.postMessage({
        type: "status",
        text: "Iniciando modelo..."
    });

    const generator =
        await pipeline(
            "text-generation",
            "onnx-community/LFM2.5-350M-ONNX",
            {
                device: "wasm",
                dtype: "q4"
            }
        );

    self.postMessage({
        type: "status",
        text: "MODELO CARREGADO"
    });

} catch (error) {

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

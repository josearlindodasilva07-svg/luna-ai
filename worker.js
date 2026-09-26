self.postMessage({
    type: "status",
    text: "Worker iniciado"
});

try {

    self.postMessage({
        type: "status",
        text: "Importando Transformers.js..."
    });

    const {
        pipeline,
        env
    } = await import(
        "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0"
    );

    self.postMessage({
        type: "status",
        text: "Transformers.js carregado"
    });

    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;

    self.postMessage({
        type: "status",
        text: "Iniciando modelo..."
    });

    const MODEL =
        "onnx-community/LFM2.5-350M-ONNX";

    const generator =
        await pipeline(
            "text-generation",
            MODEL,
            {
                device: "wasm",
                dtype: "q8"
            }
        );

    self.postMessage({
        type: "status",
        text: "MODELO CARREGADO"
    });

    self.postMessage({
        type: "loaded"
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

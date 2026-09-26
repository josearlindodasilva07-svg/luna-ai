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

    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;

    self.postMessage({
        type: "status",
        text: "Transformers.js carregado"
    });

    self.postMessage({
        type: "status",
        text: "Carregando LFM2..."
    });

    const generator =
        await pipeline(
            "text-generation",
            "onnx-community/LFM2-350M-ONNX",
            {
                device: "wasm",
                dtype: "fp16"
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

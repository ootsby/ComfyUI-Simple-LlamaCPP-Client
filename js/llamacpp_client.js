import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function getServerUrl(node) {
    return getWidget(node, "server_url")?.value ?? "http://127.0.0.1:8888";
}

async function fetchModels(serverUrl) {
    const resp = await api.fetchApi(`/llamacpp/models?url=${encodeURIComponent(serverUrl)}`);
    return await resp.json();
}

async function fetchRunning(serverUrl) {
    const resp = await api.fetchApi(`/llamacpp/running?url=${encodeURIComponent(serverUrl)}`);
    return await resp.json();
}

async function doUnload(serverUrl, model) {
    const resp = await api.fetchApi(`/llamacpp/unload?url=${encodeURIComponent(serverUrl)}&model=${encodeURIComponent(model)}`);
    return await resp.json();
}

async function doLoad(serverUrl, model) {
    const resp = await api.fetchApi(`/llamacpp/load?url=${encodeURIComponent(serverUrl)}&model=${encodeURIComponent(model)}`);
    return await resp.json();
}

function toast(msg, color = "#333") {
    const el = document.createElement("div");
    el.textContent = msg;
    Object.assign(el.style, {
        position: "fixed", bottom: "30px", right: "30px",
        background: color, color: "#fff",
        padding: "8px 16px", borderRadius: "8px",
        fontSize: "13px", zIndex: "9999",
        boxShadow: "0 2px 8px rgba(0,0,0,.4)",
        transition: "opacity .5s",
    });
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 600); }, 2800);
}

function styledBtn(label, title, color = "#3a7bd5") {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.title = title;
    Object.assign(btn.style, {
        padding: "4px 10px", margin: "2px",
        borderRadius: "5px", border: "none",
        background: color, color: "#fff",
        cursor: "pointer", fontSize: "12px",
        fontWeight: "bold", lineHeight: "1.4",
    });
    return btn;
}

function showModelPicker(models, anchorEl, onSelect) {
    document.getElementById("llamacpp_model_picker")?.remove();

    const picker = document.createElement("div");
    picker.id = "llamacpp_model_picker";
    const rect = anchorEl.getBoundingClientRect();
    Object.assign(picker.style, {
        position: "fixed",
        top: `${rect.bottom + 4}px`,
        left: `${rect.left}px`,
        zIndex: "99999",
        background: "#1e1e2e",
        border: "1px solid #555",
        borderRadius: "8px",
        padding: "6px 0",
        minWidth: "320px",
        maxHeight: "400px",
        overflowY: "auto",
        boxShadow: "0 4px 20px rgba(0,0,0,.7)",
    });

    models.forEach(m => {
        const item = document.createElement("div");
        const statusIcon = m.status === "loaded" ? "🟢" : m.status === "loading" ? "🟡" : "⬜";
        const modalities = m.input_modalities?.length ? ` [${m.input_modalities.join(", ")}]` : "";
        item.textContent = `${statusIcon} ${m.id}${modalities}`;
        Object.assign(item.style, {
            padding: "8px 14px",
            cursor: "pointer",
            color: "#cdd6f4",
            fontSize: "13px",
            whiteSpace: "nowrap",
        });
        item.addEventListener("mouseenter", () => item.style.background = "#313244");
        item.addEventListener("mouseleave", () => item.style.background = "");
        item.addEventListener("click", () => { onSelect(m.id); picker.remove(); });
        picker.appendChild(item);
    });

    document.body.appendChild(picker);

    setTimeout(() => {
        document.addEventListener("click", function handler(e) {
            if (!picker.contains(e.target)) {
                picker.remove();
                document.removeEventListener("click", handler);
            }
        });
    }, 0);
}

app.registerExtension({
    name: "LlamaCpp.RouterClient",

    async nodeCreated(node) {
        const isClient = node.comfyClass === "LlamaCppRouterClient";
        const isSelector = node.comfyClass === "LlamaCppModelSelector";
        if (!isClient && !isSelector) return;

        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "flex", flexWrap: "wrap",
            padding: "4px 6px", gap: "2px",
        });

        const btnFetch = styledBtn("🔄 Fetch Models", "Fetch model list and pick one");
        btnFetch.addEventListener("click", async () => {
            btnFetch.textContent = "⏳…";
            btnFetch.disabled = true;
            try {
                const url = getServerUrl(node);
                const data = await fetchModels(url);
                if (data.status === "ok" && data.models.length > 0) {
                    showModelPicker(data.models, btnFetch, (selected) => {
                        const w = getWidget(node, "model");
                        if (w) {
                            w.value = selected;
                            node.setDirtyCanvas(true, true);
                        }
                        toast(`✅ Model set: ${selected}`, "#2e7d32");
                    });
                } else {
                    toast(`⚠️ ${data.error ?? "No models returned by server"}`, "#b71c1c");
                }
            } catch (e) {
                toast(`❌ ${e}`, "#b71c1c");
            } finally {
                btnFetch.textContent = "🔄 Fetch Models";
                btnFetch.disabled = false;
            }
        });
        bar.appendChild(btnFetch);

        if (isClient) {
            const btnRunning = styledBtn("📋 Running", "Show currently loaded models", "#5c6bc0");
            btnRunning.addEventListener("click", async () => {
                try {
                    const data = await fetchRunning(getServerUrl(node));
                    const list = data.running ?? [];
                    toast(list.length ? `🟢 Running: ${list.join(", ")}` : "⬜ No models loaded", "#37474f");
                } catch (e) { toast(`❌ ${e}`, "#b71c1c"); }
            });
            bar.appendChild(btnRunning);

            const btnLoad = styledBtn("⬆ Load", "Load the selected model now", "#2e7d32");
            btnLoad.addEventListener("click", async () => {
                const model = getWidget(node, "model")?.value;
                if (!model) { toast("⚠️ No model selected", "#b71c1c"); return; }
                try {
                    const data = await doLoad(getServerUrl(node), model);
                    toast(data.status === "ok" ? `✅ Loading: ${model}` : `⚠️ ${data.error}`, data.status === "ok" ? "#2e7d32" : "#b71c1c");
                } catch (e) { toast(`❌ ${e}`, "#b71c1c"); }
            });
            bar.appendChild(btnLoad);

            const btnUnload = styledBtn("⏏ Unload", "Unload the selected model from GPU", "#c62828");
            btnUnload.addEventListener("click", async () => {
                const model = getWidget(node, "model")?.value;
                if (!model) { toast("⚠️ No model selected", "#b71c1c"); return; }
                if (!confirm(`Unload model "${model}"?`)) return;
                try {
                    const data = await doUnload(getServerUrl(node), model);
                    toast(data.status === "ok" ? `✅ Unloaded: ${model}` : `⚠️ ${data.error}`, data.status === "ok" ? "#2e7d32" : "#b71c1c");
                } catch (e) { toast(`❌ ${e}`, "#b71c1c"); }
            });
            bar.appendChild(btnUnload);
        }

        node.addDOMWidget("llamacpp_controls", "btn_bar", bar, {
            serialize: false,
            hideOnZoom: false,
        });
    },
});

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileMeta = document.getElementById('fileMeta');
const removeFile = document.getElementById('removeFile');
const fileList = document.getElementById('fileList');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const batchSize = document.getElementById('batchSize');
const temperature = document.getElementById('temperature');
const topP = document.getElementById('topP');
const maxTokens = document.getElementById('maxTokens');
const llmStatusBadge = document.getElementById('llmStatusBadge');
const translateBtn = document.getElementById('translateBtn');
const progressPanel = document.getElementById('progressPanel');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const statusText = document.getElementById('statusText');
const batchInfo = document.getElementById('batchInfo');
const elapsedTime = document.getElementById('elapsedTime');
const speedInfo = document.getElementById('speedInfo');
const linesInfo = document.getElementById('linesInfo');
const etaInfo = document.getElementById('etaInfo');
const downloadBtn = document.getElementById('downloadBtn');
const cancelBtn = document.getElementById('cancelBtn');
const glossaryList = document.getElementById('glossaryList');
const glossarySource = document.getElementById('glossarySource');
const glossaryTarget = document.getElementById('glossaryTarget');
const addGlossaryBtn = document.getElementById('addGlossaryBtn');
const historyToggle = document.getElementById('historyToggle');
const historyList = document.getElementById('historyList');

let selectedFiles = [];
let currentJobId = null;
let glossaryEntries = [];

function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
}

// LANG_NAMES maps ISO codes -> names for history entries saved before the LLM
// migration (new entries store the language NAME directly). Populated from the
// API on load so it always matches the backend list.
const LANG_NAMES = {};

function langName(code) {
    if (LANG_NAMES[code]) return LANG_NAMES[code];
    // Name-based entry (e.g. "Javanese", "English"): show as stored
    if (code && code.length > 3) return code;
    return code ? code.toUpperCase() : '';
}

async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        const entries = await resp.json();
        renderHistory(entries);
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderHistory(entries) {
    historyList.innerHTML = '';
    if (!entries || entries.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No translations yet</div>';
        return;
    }

    entries.forEach(entry => {
        const div = document.createElement('div');
        div.className = 'history-entry';

        const fileDisplay = entry.filenames.length > 2
            ? `${entry.filenames[0]} +${entry.filenames.length - 1} more`
            : entry.filenames.join(', ');

        const date = new Date(entry.completed_at);
        const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
        const timeStr = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

        div.innerHTML = `
            <div class="history-entry-header">
                <span class="history-filenames">${esc(fileDisplay)}</span>
                ${entry.is_batch ? '<span class="history-badge batch">Batch</span>' : ''}
            </div>
            <div class="history-meta">
                <span>${langName(entry.source_lang)} → ${langName(entry.target_lang)}</span>
                <span>${entry.total_lines} lines</span>
                <span>${formatTime(entry.elapsed_seconds)}</span>
                <span>${entry.lines_per_second} lines/s</span>
                <span>${dateStr} ${timeStr}</span>
            </div>
            <div class="history-actions">
                <button class="btn-small btn-download" data-job="${entry.job_id}" data-batch="${entry.is_batch}">Download</button>
                <button class="btn-small btn-delete-history" data-job="${entry.job_id}">Delete</button>
            </div>
        `;
        historyList.appendChild(div);
    });

    historyList.querySelectorAll('.btn-download').forEach(btn => {
        btn.addEventListener('click', () => {
            window.location.href = `/api/history/${btn.dataset.job}/download`;
        });
    });

    historyList.querySelectorAll('.btn-delete-history').forEach(btn => {
        btn.addEventListener('click', async () => {
            if (!confirm('Delete this translation from history?')) return;
            try {
                await fetch(`/api/history/${btn.dataset.job}`, { method: 'DELETE' });
                loadHistory();
            } catch (e) {
                console.error('Delete error:', e);
            }
        });
    });
}

async function loadLLMStatus() {
    try {
        const resp = await fetch('/api/llm-status');
        const info = await resp.json();
        if (info.connected) {
            const name = info.model || 'local LLM';
            llmStatusBadge.textContent = 'Connected: ' + name;
            llmStatusBadge.className = 'device-badge gpu';
            llmStatusBadge.title = info.base_url;
        } else {
            llmStatusBadge.textContent = 'LLM Offline';
            llmStatusBadge.className = 'device-badge offline';
            llmStatusBadge.title = info.error || 'Cannot reach llama-server';
        }
    } catch (e) {
        console.error('Failed to load LLM status:', e);
        llmStatusBadge.textContent = 'LLM Offline';
        llmStatusBadge.className = 'device-badge offline';
        llmStatusBadge.title = 'Cannot reach llama-server';
    }
}

async function loadLanguages() {
    try {
        const resp = await fetch('/api/languages');
        const langs = await resp.json();
        langs.forEach(l => {
            // Value = the full language NAME: sent verbatim to the LLM prompt
            // ("Translate from English to Javanese") — no code validation needed.
            sourceLang.add(new Option(l.name, l.name));
            targetLang.add(new Option(l.name, l.name));
        });
        sourceLang.value = 'English';
        targetLang.value = 'Indonesian';
        // Populate code->name map (for legacy code-based history entries)
        langs.forEach(l => { LANG_NAMES[l.code] = l.name; });
    } catch (e) {
        console.error('Failed to load languages:', e);
    }
}

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    addFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    addFiles(e.target.files);
    fileInput.value = '';
});

function addFiles(fileListInput) {
    for (const file of fileListInput) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['srt', 'ass', 'ssa', 'vtt'].includes(ext)) {
            alert(`Unsupported format: ${file.name}. Use SRT, ASS, SSA, or VTT.`);
            continue;
        }
        const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
        if (!exists) {
            selectedFiles.push(file);
        }
    }
    renderFileList();
}

function removeFileByIndex(idx) {
    selectedFiles.splice(idx, 1);
    renderFileList();
}

function renderFileList() {
    fileList.innerHTML = '';
    if (selectedFiles.length === 0) {
        fileInfo.style.display = 'none';
        dropZone.style.display = 'block';
        translateBtn.disabled = true;
        return;
    }

    dropZone.style.display = 'none';
    translateBtn.disabled = false;

    if (selectedFiles.length === 1) {
        const f = selectedFiles[0];
        const ext = f.name.split('.').pop().toUpperCase();
        fileInfo.style.display = 'flex';
        fileName.textContent = f.name;
        fileMeta.textContent = `${ext} \u2022 ${(f.size / 1024).toFixed(1)} KB`;
        fileList.style.display = 'none';
        return;
    }

    fileInfo.style.display = 'none';
    fileList.style.display = 'block';

    const totalSize = selectedFiles.reduce((a, f) => a + f.size, 0);
    const header = document.createElement('div');
    header.className = 'file-list-header';
    header.innerHTML = `
        <span>${selectedFiles.length} files selected (${(totalSize / 1024).toFixed(1)} KB total)</span>
        <button class="btn-small btn-remove" onclick="clearAllFiles()">Clear All</button>
    `;
    fileList.appendChild(header);

    selectedFiles.forEach((f, i) => {
        const ext = f.name.split('.').pop().toUpperCase();
        const div = document.createElement('div');
        div.className = 'file-list-item';
        div.innerHTML = `
            <span class="file-list-name">${esc(f.name)}</span>
            <span class="file-list-meta">${ext} \u2022 ${(f.size / 1024).toFixed(1)} KB</span>
            <button class="btn-small btn-remove" data-idx="${i}">\u2715</button>
        `;
        fileList.appendChild(div);
    });

    fileList.querySelectorAll('.btn-remove[data-idx]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeFileByIndex(parseInt(btn.dataset.idx));
        });
    });
}

removeFile.addEventListener('click', () => {
    selectedFiles = [];
    renderFileList();
});

window.clearAllFiles = function() {
    selectedFiles = [];
    renderFileList();
};

addGlossaryBtn.addEventListener('click', () => {
    const src = glossarySource.value.trim();
    const tgt = glossaryTarget.value.trim();
    if (!src || !tgt) return;

    glossaryEntries.push({ source: src, target: tgt, case_sensitive: true });
    renderGlossary();
    glossarySource.value = '';
    glossaryTarget.value = '';
});

function renderGlossary() {
    glossaryList.innerHTML = '';
    glossaryEntries.forEach((entry, i) => {
        const div = document.createElement('div');
        div.className = 'glossary-item';
        div.innerHTML = `
            <span class="source">${esc(entry.source)}</span>
            <span class="arrow">\u2192</span>
            <span class="target">${esc(entry.target)}</span>
            <button class="remove-btn" data-idx="${i}">\u2715</button>
        `;
        glossaryList.appendChild(div);
    });

    glossaryList.querySelectorAll('.remove-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            glossaryEntries.splice(parseInt(btn.dataset.idx), 1);
            renderGlossary();
        });
    });
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

translateBtn.addEventListener('click', startTranslation);

async function startTranslation() {
    if (selectedFiles.length === 0) return;

    translateBtn.disabled = true;
    translateBtn.textContent = 'Starting...';
    progressPanel.style.display = 'block';
    downloadBtn.style.display = 'none';
    cancelBtn.style.display = 'block';
    cancelBtn.disabled = false;
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    statusText.textContent = 'Uploading...';
    batchInfo.textContent = '0 / 0';
    linesInfo.textContent = '0 / 0';
    elapsedTime.textContent = '0s';
    speedInfo.textContent = '- lines/s';
    etaInfo.textContent = '-';

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));
    formData.append('source_lang', sourceLang.value);
    formData.append('target_lang', targetLang.value);
    formData.append('batch_size', batchSize.value);
    formData.append('temperature', temperature.value);
    formData.append('top_p', topP.value);
    formData.append('max_tokens', maxTokens.value);
    formData.append('glossary', JSON.stringify(glossaryEntries));

    const endpoint = '/api/translate-batch';

    try {
        const resp = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || 'Translation failed');
            translateBtn.disabled = false;
            translateBtn.textContent = 'Translate';
            return;
        }

        const data = await resp.json();
        currentJobId = data.job_id;
        pollProgress();
    } catch (e) {
        alert('Error: ' + e.message);
        translateBtn.disabled = false;
        translateBtn.textContent = 'Translate';
    }
}

function pollProgress() {
    if (!currentJobId) return;

    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/progress/${currentJobId}`);
            const data = await resp.json();

            const pct = Math.round(data.percent);
            progressBar.style.width = pct + '%';
            progressText.textContent = pct + '%';
            statusText.textContent = data.status;
            batchInfo.textContent = `${data.completed_batches} / ${data.total_batches}`;
            linesInfo.textContent = `${data.completed_lines} / ${data.total_lines}`;
            elapsedTime.textContent = formatTime(data.elapsed_seconds);
            speedInfo.textContent = data.lines_per_second > 0 ? `${data.lines_per_second} lines/s` : '- lines/s';
            etaInfo.textContent = data.eta_seconds > 0 ? formatTime(data.eta_seconds) : '-';

            if (data.status === 'completed') {
                clearInterval(interval);
                const count = selectedFiles.length;
                downloadBtn.textContent = count > 1 ? `Download All (${count} files ZIP)` : 'Download Translated File';
                downloadBtn.style.display = 'block';
                cancelBtn.style.display = 'none';
                translateBtn.disabled = false;
                translateBtn.textContent = 'Translate';
                loadHistory();
            } else if (data.status === 'cancelled') {
                clearInterval(interval);
                statusText.textContent = 'Cancelled';
                cancelBtn.style.display = 'none';
                translateBtn.disabled = false;
                translateBtn.textContent = 'Translate';
            } else if (data.status === 'failed') {
                clearInterval(interval);
                statusText.textContent = 'Failed: ' + (data.error || 'Unknown error');
                translateBtn.disabled = false;
                translateBtn.textContent = 'Translate';
            }
        } catch (e) {
            console.error('Poll error:', e);
        }
    }, 800);
}

downloadBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/api/download-batch/${currentJobId}`;
    }
});

cancelBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Cancelling...';
    try {
        await fetch(`/api/cancel/${currentJobId}`, { method: 'POST' });
    } catch (e) {
        console.error('Cancel error:', e);
    }
});

loadLanguages();
loadLLMStatus();
loadHistory();

document.querySelector('.guide-toggle').addEventListener('click', () => {
    const content = document.getElementById('guideContent');
    const icon = document.querySelector('.toggle-icon');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '\u25BE';
    } else {
        content.style.display = 'none';
        icon.textContent = '\u25B8';
    }
});

historyToggle.addEventListener('click', () => {
    const icon = historyToggle.querySelector('.toggle-icon');
    if (historyList.style.display === 'none') {
        historyList.style.display = 'block';
        icon.textContent = '\u25BE';
        loadHistory();
    } else {
        historyList.style.display = 'none';
        icon.textContent = '\u25B8';
    }
});

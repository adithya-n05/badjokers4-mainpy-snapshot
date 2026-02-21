const $ = (id) => document.getElementById(id);

const els = {
  routerDot: $("routerDot"),
  routerText: $("routerText"),
  voiceDot: $("voiceDot"),
  voiceText: $("voiceText"),
  toolCount: $("toolCount"),
  sampleWrap: $("sampleWrap"),
  textInput: $("textInput"),
  sendTextBtn: $("sendTextBtn"),
  recordBtn: $("recordBtn"),
  recordState: $("recordState"),
  summaryText: $("summaryText"),
  transcriptBox: $("transcriptBox"),
  feedbackList: $("feedbackList"),
  kPatients: $("kPatients"),
  kQueue: $("kQueue"),
  kReferrals: $("kReferrals"),
  kFollowups: $("kFollowups"),
  activityBox: $("activityBox"),
};

const samples = [
  "Register patient Amina with severe wheeze in Zone B, assign red triage, notify Dr. Khan, and set 20 minute follow-up.",
  "Check oxygen cylinder inventory, request urgent restock of 15, and broadcast alert to respiratory team.",
  "Create referral for Musa to Regional Hospital for persistent low oxygen and dispatch ambulance.",
  "Record vitals for Layla heart rate 130 oxygen 88 and log critical incident for respiratory surge.",
];

let isRecording = false;
let mediaStream = null;
let audioCtx = null;
let sourceNode = null;
let processorNode = null;
let recordedChunks = [];
let recordedSampleRate = 16000;
let isProcessingVoice = false;

function setDot(dotEl, ok) {
  dotEl.className = `dot ${ok ? "ok" : "warn"}`;
}

function b64FromArrayBuffer(buffer) {
  const bytes = new Uint8Array(buffer);
  let bin = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    const slice = bytes.subarray(i, i + step);
    bin += String.fromCharCode(...slice);
  }
  return btoa(bin);
}

function mergeFloat32(chunks) {
  const total = chunks.reduce((acc, arr) => acc + arr.length, 0);
  const result = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function encodeWav(floatData, sampleRate) {
  const buffer = new ArrayBuffer(44 + floatData.length * 2);
  const view = new DataView(buffer);

  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(off + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + floatData.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, floatData.length * 2, true);

  let offset = 44;
  for (let i = 0; i < floatData.length; i += 1) {
    let s = Math.max(-1, Math.min(1, floatData[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, s, true);
    offset += 2;
  }
  return buffer;
}

function renderState(state) {
  els.kPatients.textContent = String(state.patient_count ?? 0);
  els.kQueue.textContent = String(state.triage_queue_count ?? 0);
  els.kReferrals.textContent = String(state.referral_count ?? 0);
  els.kFollowups.textContent = String(state.followup_count ?? 0);
  els.activityBox.textContent = JSON.stringify(state.latest_activities ?? [], null, 2);
}

function renderOutcome(payload) {
  els.summaryText.textContent = payload.summary || "Completed.";

  if (payload.transcript) {
    els.transcriptBox.classList.remove("hidden");
    els.transcriptBox.textContent = `Transcript: ${payload.transcript}`;
    els.textInput.value = payload.transcript;
  } else {
    els.transcriptBox.classList.add("hidden");
    els.transcriptBox.textContent = "";
  }

  els.feedbackList.innerHTML = "";
  for (const outcome of payload.outcomes || []) {
    const card = document.createElement("article");
    card.className = `feedback-item ${outcome.success ? "ok" : "fail"}`;
    card.innerHTML = `
      <div class="feedback-head">
        <span class="feedback-name">${outcome.name}</span>
        <span class="tag ${outcome.success ? "ok" : "fail"}">${outcome.success ? "DONE" : "FAILED"}</span>
      </div>
      <div>${outcome.message}</div>
    `;
    els.feedbackList.appendChild(card);
  }

  if (payload.state) {
    renderState(payload.state);
  }
}

async function callApi(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok || !data.ok) {
    throw new Error(data.error || `Request failed (${resp.status})`);
  }
  return data;
}

function setBusy(busy, label = "Run Command") {
  els.sendTextBtn.disabled = busy;
  if (!isRecording) {
    els.recordBtn.disabled = busy || isProcessingVoice;
  }
  els.sendTextBtn.textContent = busy ? label : "Run Command";
}

async function submitText() {
  const text = (els.textInput.value || "").trim();
  if (!text) {
    alert("Please enter a command first.");
    return;
  }
  setBusy(true, "Running...");
  try {
    const data = await callApi("/api/text-command", { text });
    renderOutcome(data);
  } catch (err) {
    alert(err.message);
  } finally {
    setBusy(false);
  }
}

async function submitAudioB64(audioB64, sourceLabel) {
  isProcessingVoice = true;
  els.recordBtn.disabled = true;
  setBusy(true, sourceLabel === "mic" ? "Transcribing..." : "Processing voice...");
  try {
    const data = await callApi("/api/voice-command", { audio_b64: audioB64 });
    renderOutcome(data);
  } catch (err) {
    alert(err.message);
  } finally {
    isProcessingVoice = false;
    setBusy(false);
    if (!isRecording) els.recordBtn.disabled = false;
  }
}

async function cleanupRecording() {
  try {
    if (processorNode) {
      processorNode.disconnect();
      processorNode.onaudioprocess = null;
    }
  } catch (_) {}
  try {
    if (sourceNode) sourceNode.disconnect();
  } catch (_) {}
  try {
    if (mediaStream) {
      for (const t of mediaStream.getTracks()) t.stop();
    }
  } catch (_) {}
  try {
    if (audioCtx && audioCtx.state !== "closed") {
      await audioCtx.close();
    }
  } catch (_) {}
  processorNode = null;
  sourceNode = null;
  mediaStream = null;
  audioCtx = null;
}

async function startRecording() {
  await cleanupRecording();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  recordedSampleRate = audioCtx.sampleRate || 16000;
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  recordedChunks = [];
  processorNode.onaudioprocess = (e) => {
    const channel = e.inputBuffer.getChannelData(0);
    recordedChunks.push(new Float32Array(channel));
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);
}

async function stopRecordingAndSubmit() {
  try {
    const merged = mergeFloat32(recordedChunks);
    if (!merged.length) {
      throw new Error("No audio captured. Please try recording again.");
    }
    const durationSec = merged.length / Math.max(1, recordedSampleRate);
    if (durationSec < 0.45) {
      throw new Error("Recording too short. Please speak for at least half a second.");
    }
    const wav = encodeWav(merged, recordedSampleRate);
    const b64 = b64FromArrayBuffer(wav);
    await submitAudioB64(b64, "mic");
  } finally {
    await cleanupRecording();
    recordedChunks = [];
  }
}

async function toggleRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Microphone recording is not supported in this browser.");
    return;
  }
  if (isProcessingVoice) {
    return;
  }

  if (!isRecording) {
    try {
      await startRecording();
      isRecording = true;
      els.recordBtn.textContent = "Stop & Send";
      els.recordState.textContent = "Recording...";
      els.recordBtn.disabled = false;
    } catch (err) {
      alert(`Could not start recording: ${err.message}`);
    }
    return;
  }

  isRecording = false;
  els.recordBtn.textContent = "Start Recording";
  els.recordState.textContent = "Processing voice...";
  try {
    await stopRecordingAndSubmit();
    els.recordState.textContent = "Voice command processed.";
  } catch (err) {
    els.recordState.textContent = "Voice processing failed.";
    alert(err.message);
  } finally {
    els.recordBtn.textContent = "Start Recording";
    if (!isProcessingVoice) els.recordBtn.disabled = false;
  }
}

function setupSamples() {
  samples.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample";
    btn.textContent = s;
    btn.addEventListener("click", () => {
      els.textInput.value = s;
      els.textInput.focus();
    });
    els.sampleWrap.appendChild(btn);
  });
}

async function refreshHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || "Health check failed");

    const routerReady = !!data.router?.ready;
    const voiceReady = !!data.transcription?.ready;
    setDot(els.routerDot, routerReady);
    setDot(els.voiceDot, voiceReady);
    els.routerText.textContent = routerReady ? "Router connected" : "Router unavailable";
    els.voiceText.textContent = voiceReady ? "Voice transcription ready" : "Voice transcription unavailable";
    els.toolCount.textContent = `Tools available: ${data.tool_count ?? "-"}`;

    if (!voiceReady) {
      if (!isRecording) els.recordBtn.disabled = true;
      els.recordState.textContent = "Voice unavailable. Use text mode or fix Cactus setup.";
    } else if (!isRecording) {
      els.recordBtn.disabled = false;
      els.recordState.textContent = "Not recording";
    }

    if (data.state) renderState(data.state);
  } catch (err) {
    setDot(els.routerDot, false);
    setDot(els.voiceDot, false);
    els.routerText.textContent = "Backend offline";
    els.voiceText.textContent = "Backend offline";
    els.toolCount.textContent = "Tools: -";
  }
}

function bindEvents() {
  els.sendTextBtn.addEventListener("click", submitText);
  els.recordBtn.addEventListener("click", toggleRecording);
}

function boot() {
  setupSamples();
  bindEvents();
  refreshHealth();
  window.setInterval(refreshHealth, 5000);
}

boot();

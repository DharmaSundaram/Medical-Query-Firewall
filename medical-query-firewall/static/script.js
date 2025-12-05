// static/script.js
const logEl = document.getElementById("log");
const msg = document.getElementById("msg");
const send = document.getElementById("send");
const sample = document.getElementById("sample");

function appendLine(html) {
  logEl.innerHTML += html + "<hr/>";
  logEl.scrollTop = logEl.scrollHeight;
}

send.onclick = async () => {
  const text = msg.value.trim();
  if (!text) return;
  appendLine(`<b>You:</b> ${text}`);
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });
  const j = await resp.json();
  if (j.decision === "BLOCK") {
    appendLine(`<b class="block">Decision: BLOCK</b><div>${j.safe_response}</div><pre>${JSON.stringify(j.explain, null, 2)}</pre>`);
  } else if (j.decision === "WARN") {
    appendLine(`<b class="warn">Decision: WARN</b><div>${j.warning}</div><div>${j.llm_response}</div><pre>${JSON.stringify(j.explain, null, 2)}</pre>`);
  } else {
    appendLine(`<b class="allow">Decision: ALLOW</b><div>${j.llm_response}</div><pre>${JSON.stringify(j.explain, null, 2)}</pre>`);
  }
};

sample.onclick = () => {
  const options = [
    "What are symptoms of flu?",
    "Can I get antibiotics without prescription?",
    "How to make a saline IV at home?",
    "My Aadhaar number is 123412341234",
    "What is a normal blood pressure?"
  ];
  msg.value = options[Math.floor(Math.random()*options.length)];
};
// ================================
// ADMIN AUDIT DOWNLOAD LOGIC
// ================================

// Button & message area
const downloadBtn = document.getElementById("download-audit");
const adminMsg = document.getElementById("admin-msg");

// Replace with your real admin key:
const ADMIN_KEY = "hackathon-demo-key-CHANGE_ME";

downloadBtn.onclick = async () => {
    adminMsg.innerText = "";  // clear previous message

    try {
        const resp = await fetch("/admin/audit?limit=500", {
            method: "GET",
            headers: {
                "x-api-key": ADMIN_KEY
            }
        });

        if (!resp.ok) {
            adminMsg.style.color = "#d32f2f";
            adminMsg.innerText = `Error: ${resp.status} ${resp.statusText}`;
            return;
        }

        const data = await resp.json();

        // Convert JSON to file
        const blob = new Blob(
            [JSON.stringify(data, null, 2)],
            { type: "application/json" }
        );

        // Create temporary link to trigger download
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "audit_logs.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);

        // Success message
        adminMsg.style.color = "#2e7d32";
        adminMsg.innerText = "✔ Audit logs downloaded successfully!";

    } catch (error) {
        adminMsg.style.color = "#d32f2f";
        adminMsg.innerText = "Error fetching audit logs: " + error;
    }
};

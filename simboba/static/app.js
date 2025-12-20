// Simboba Frontend Application

const API_BASE = '/api';

// State
let datasets = [];
let runs = [];
let evals = [];
let evalErrors = [];
let settings = {};
let currentDataset = null;
let uploadedFiles = [];
let generatedCases = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupDragAndDrop();
    loadInitialData();
});

// Navigation
function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            const page = link.dataset.page;
            navigateTo(page);
        });
    });

    // Handle browser back/forward
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();
}

function handleHashChange() {
    const hash = window.location.hash.slice(1) || 'dashboard';
    const parts = hash.split('/');
    const page = parts[0];
    const id = parts[1];

    if (page === 'datasets' && id) {
        navigateTo('datasets');
        loadDatasetDetail(parseInt(id));
    } else if (page === 'runs' && id) {
        // Open run sidebar
        navigateTo('runs');
        openRunSidebar(parseInt(id));
    } else {
        navigateTo(page);
    }
}

function navigateTo(page) {
    // Close any open sidebars
    if (selectedCaseId !== null) closeCaseSidebar();
    if (selectedRunId !== null) closeRunSidebar();

    // Update nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelector(`.nav-link[data-page="${page}"]`)?.classList.add('active');

    // Update pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`)?.classList.add('active');

    // Load content
    if (page === 'dashboard') renderDashboard();
    if (page === 'datasets') renderDatasetsPage();
    if (page === 'runs') renderRunsPage();
    if (page === 'playground') renderPlaygroundPage();
    if (page === 'settings') renderSettingsPage();

    // Reset detail views
    currentDataset = null;
}

// Data Loading
async function loadInitialData() {
    await Promise.all([loadDatasets(), loadRuns(), loadEvals(), loadEvalErrors(), loadSettings()]);
    showEvalErrorsBanner();
    handleHashChange();
}

async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/datasets`, { cache: 'no-store' });
        datasets = await response.json();
    } catch (e) {
        console.error('Failed to load datasets:', e);
    }
}

async function loadRuns() {
    try {
        const response = await fetch(`${API_BASE}/runs`);
        runs = await response.json();
    } catch (e) {
        console.error('Failed to load runs:', e);
    }
}

async function loadEvals() {
    try {
        const response = await fetch(`${API_BASE}/evals`);
        evals = await response.json();
    } catch (e) {
        console.error('Failed to load evals:', e);
    }
}

async function loadEvalErrors() {
    try {
        const response = await fetch(`${API_BASE}/evals/errors`);
        evalErrors = await response.json();
    } catch (e) {
        console.error('Failed to load eval errors:', e);
    }
}

function showEvalErrorsBanner() {
    // Remove existing banner if any
    const existing = document.getElementById('eval-errors-banner');
    if (existing) existing.remove();

    if (evalErrors.length === 0) return;

    const banner = document.createElement('div');
    banner.id = 'eval-errors-banner';
    banner.style.cssText = `
        background: var(--red-50);
        border-bottom: 1px solid var(--red-500);
        padding: 12px 32px;
        display: flex;
        align-items: center;
        gap: 12px;
    `;

    const errorList = evalErrors.map(e => `${e.file}: ${e.error}`).join('\n');
    banner.innerHTML = `
        <span class="status-dot fail"></span>
        <span style="color: #991b1b; font-size: 13px;">
            <strong>${evalErrors.length} eval file(s) failed to load.</strong>
            Run <code style="background: white; padding: 2px 6px; border-radius: 3px;">boba evals</code> for details.
        </span>
        <button onclick="this.parentElement.remove()" style="margin-left: auto; background: none; border: none; color: #991b1b; cursor: pointer; font-size: 18px;">&times;</button>
    `;

    // Insert after header
    const header = document.querySelector('header');
    header.after(banner);
}

async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        settings = await response.json();
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

async function saveSettings(updates) {
    try {
        const response = await fetch(`${API_BASE}/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });
        settings = await response.json();
        return true;
    } catch (e) {
        console.error('Failed to save settings:', e);
        return false;
    }
}

// Dashboard
function renderDashboard() {
    const container = document.getElementById('dashboard-content');

    if (datasets.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>Get started</h3>
                <p>Create your first dataset to start running evaluations.</p>
                <button class="btn btn-primary" onclick="navigateTo('datasets'); showNewDatasetModal();">Create Dataset</button>
            </div>
        `;
        return;
    }

    if (runs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>Ready to run</h3>
                <p>You have ${datasets.length} dataset${datasets.length > 1 ? 's' : ''}. Run your first evaluation.</p>
                <button class="btn btn-primary" onclick="navigateTo('datasets')">Go to Datasets</button>
            </div>
        `;
        return;
    }

    const recentRuns = runs.slice(0, 10);
    container.innerHTML = `
        <div class="page-header">
            <div>
                <h1 class="page-title">Recent Runs</h1>
            </div>
        </div>
        <div class="table-container">
            <div id="dashboard-runs-list">
                ${recentRuns.map(r => renderRunListItem(r)).join('')}
            </div>
        </div>
        <p style="margin-top: 16px;">
            <a class="back-link" onclick="navigateTo('runs')" style="margin-bottom: 0;">View all runs →</a>
        </p>
    `;
}

// Datasets Page
function renderDatasetsPage() {
    const container = document.getElementById('datasets-content');

    if (currentDataset) {
        renderDatasetDetail();
        return;
    }

    if (datasets.length === 0) {
        container.innerHTML = `
            <div class="page-header">
                <div>
                    <h1 class="page-title">Datasets</h1>
                    <p class="page-subtitle">Manage your evaluation datasets</p>
                </div>
                <button class="btn btn-primary" onclick="showNewDatasetModal()">+ New Dataset</button>
            </div>
            <div class="empty-state">
                <h3>No datasets yet</h3>
                <p>Create your first dataset to start running evaluations.</p>
                <button class="btn btn-primary" onclick="showNewDatasetModal()">+ New Dataset</button>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="page-header">
            <div>
                <h1 class="page-title">Datasets</h1>
                <p class="page-subtitle">Manage your evaluation datasets</p>
            </div>
            <button class="btn btn-primary" onclick="showNewDatasetModal()">+ New Dataset</button>
        </div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Cases</th>
                        <th>Updated</th>
                        <th style="text-align: right;">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${datasets.map(d => `
                        <tr onclick="viewDataset(${d.id})">
                            <td>
                                <div class="cell-main">${escapeHtml(d.name)}</div>
                                ${d.description ? `<div class="cell-sub">${escapeHtml(d.description)}</div>` : ''}
                            </td>
                            <td><span class="mono">${d.case_count}</span></td>
                            <td><span style="color: var(--zinc-500);">${relativeTime(d.updated_at)}</span></td>
                            <td class="cell-actions" onclick="event.stopPropagation();">
                                <button class="btn btn-ghost btn-sm" onclick="viewDataset(${d.id})">View</button>
                                <button class="btn btn-ghost btn-sm danger" onclick="confirmDeleteDataset(${d.id}, '${escapeHtml(d.name)}', ${d.case_count})">Delete</button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

async function viewDataset(id) {
    window.location.hash = `datasets/${id}`;
}

async function loadDatasetDetail(id) {
    try {
        const [dsRes, casesRes] = await Promise.all([
            fetch(`${API_BASE}/datasets/${id}`),
            fetch(`${API_BASE}/cases?dataset_id=${id}`)
        ]);
        currentDataset = await dsRes.json();
        currentDataset.cases = await casesRes.json();
        renderDatasetDetail();
    } catch (e) {
        showToast('Failed to load dataset', true);
    }
}

let selectedCaseId = null;
let selectedRunId = null;

function renderDatasetDetail() {
    const container = document.getElementById('datasets-content');
    const d = currentDataset;
    const cases = d.cases || [];

    container.innerHTML = `
        <a class="back-link" onclick="window.location.hash = 'datasets'">← Back to Datasets</a>
        <div class="page-header">
            <div>
                <h1 class="page-title">${escapeHtml(d.name)}</h1>
                ${d.description ? `<p class="page-subtitle">${escapeHtml(d.description)}</p>` : ''}
                <p class="meta"><span>${cases.length} cases</span></p>
            </div>
            <button class="btn btn-primary" onclick="showAddCaseModal()">+ Add Case</button>
        </div>
        ${cases.length === 0 ? `
            <div class="empty-state">
                <h3>No cases yet</h3>
                <p>Add cases manually or generate them with AI.</p>
                <button class="btn btn-primary" onclick="showAddCaseModal()">+ Add Case</button>
            </div>
        ` : `
            <div class="table-container">
                <div id="cases-list">
                    ${cases.map(c => renderCaseListItem(c)).join('')}
                </div>
            </div>
        `}
    `;
}

function renderCaseListItem(c) {
    const name = c.name || `Case #${c.id}`;
    const messageCount = c.inputs.length;

    return `
        <div class="case-list-item" onclick="openCaseSidebar(${c.id})" id="case-item-${c.id}">
            <span class="case-list-name">${escapeHtml(name)}</span>
            <span class="case-list-meta">${messageCount} message${messageCount !== 1 ? 's' : ''}</span>
            <span class="case-list-arrow">›</span>
        </div>
    `;
}

function openCaseSidebar(caseId) {
    const c = currentDataset.cases.find(x => x.id === caseId);
    if (!c) return;

    selectedCaseId = caseId;

    // Update selected state
    document.querySelectorAll('.case-list-item').forEach(el => el.classList.remove('selected'));
    document.getElementById(`case-item-${caseId}`)?.classList.add('selected');

    // Update sidebar title
    document.getElementById('sidebar-case-title').textContent = c.name || `Case #${c.id}`;

    // Render conversation and expected outcome
    const content = document.getElementById('sidebar-case-content');
    content.innerHTML = `
        <div class="sidebar-section">
            <div class="sidebar-section-label">Conversation</div>
            ${c.inputs.map(m => `
                <div class="conversation-message">
                    <span class="message-role-badge ${m.role}">${m.role}</span>
                    <div class="message-content">${escapeHtml(m.message)}</div>
                </div>
            `).join('')}
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-label">Expected Outcome</div>
            <div class="expected-outcome-box">${escapeHtml(c.expected_outcome)}</div>
            ${c.expected_source ? `
                <div class="source-ref" style="margin-top: 12px;">
                    <span class="mono">${escapeHtml(c.expected_source.file)}</span>
                    <span class="source-page">p. ${c.expected_source.page}</span>
                </div>
                ${c.expected_source.excerpt ? `<div style="margin-top: 8px; font-style: italic; color: var(--zinc-500); font-size: 13px;">"${escapeHtml(c.expected_source.excerpt)}"</div>` : ''}
            ` : ''}
        </div>
    `;

    // Wire up footer buttons
    document.getElementById('sidebar-edit-btn').onclick = () => {
        closeCaseSidebar();
        showEditCaseModal(caseId);
    };
    document.getElementById('sidebar-delete-btn').onclick = () => {
        closeCaseSidebar();
        deleteCase(caseId);
    };

    // Show sidebar
    document.getElementById('sidebar-overlay').classList.add('active');
    document.getElementById('case-sidebar').classList.add('active');
}

function closeCaseSidebar() {
    selectedCaseId = null;
    document.getElementById('sidebar-overlay').classList.remove('active');
    document.getElementById('case-sidebar').classList.remove('active');
    document.querySelectorAll('.case-list-item').forEach(el => el.classList.remove('selected'));
}

// Generic sidebar close (for overlay click)
function closeSidebar() {
    if (selectedCaseId !== null) {
        closeCaseSidebar();
    }
    if (selectedRunId !== null) {
        closeRunSidebar();
    }
}

// Close sidebar on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (selectedCaseId !== null) {
            closeCaseSidebar();
        }
        if (selectedRunId !== null) {
            closeRunSidebar();
        }
    }
});

function renderCaseRow(c) {
    const name = c.name || `Case #${c.id}`;
    const inputPreview = c.inputs.length > 0 ? truncate(c.inputs[0].message, 50) : '-';
    const outcomePreview = truncate(c.expected_outcome, 50);

    return `
        <tr onclick="toggleCaseExpand(${c.id})" id="case-row-${c.id}">
            <td><span class="expand-icon" id="expand-icon-${c.id}">›</span></td>
            <td><div class="cell-main">${escapeHtml(name)}</div></td>
            <td><span style="color: var(--zinc-600);">${escapeHtml(inputPreview)}</span></td>
            <td><span style="color: var(--zinc-600);">${escapeHtml(outcomePreview)}</span></td>
            <td class="cell-actions" onclick="event.stopPropagation();">
                <button class="btn btn-ghost btn-sm" onclick="showEditCaseModal(${c.id})">Edit</button>
                <button class="btn btn-ghost btn-sm danger" onclick="deleteCase(${c.id})">Delete</button>
            </td>
        </tr>
        <tr id="case-expand-${c.id}" style="display: none;">
            <td colspan="5" style="padding: 0;">
                <div class="expandable-content active">
                    ${renderCaseDetail(c)}
                </div>
            </td>
        </tr>
    `;
}

function renderCaseDetail(c) {
    return `
        <div class="detail-grid">
            <div>
                <div class="case-content-label">Inputs</div>
                <div class="messages-preview">
                    ${c.inputs.map(m => `
                        <div class="message-preview">
                            <span class="message-role">${m.role}</span>
                            <span>${escapeHtml(m.message)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div>
                <div class="case-content-label">Expected Outcome</div>
                <div>${escapeHtml(c.expected_outcome)}</div>
                ${c.expected_source ? `
                    <div class="source-ref">
                        <span class="mono">${escapeHtml(c.expected_source.file)}</span>
                        <span class="source-page">p. ${c.expected_source.page}</span>
                    </div>
                    ${c.expected_source.excerpt ? `<div style="margin-top: 8px; font-style: italic; color: var(--zinc-500); font-size: 12px;">"${escapeHtml(c.expected_source.excerpt)}"</div>` : ''}
                ` : ''}
            </div>
        </div>
    `;
}

function toggleCaseExpand(id) {
    const expandRow = document.getElementById(`case-expand-${id}`);
    const icon = document.getElementById(`expand-icon-${id}`);

    if (expandRow.style.display === 'none') {
        expandRow.style.display = 'table-row';
        icon.classList.add('expanded');
    } else {
        expandRow.style.display = 'none';
        icon.classList.remove('expanded');
    }
}

// Runs Page
function renderRunsPage() {
    const container = document.getElementById('runs-content');

    if (runs.length === 0) {
        container.innerHTML = `
            <div class="page-header">
                <div>
                    <h1 class="page-title">Runs</h1>
                    <p class="page-subtitle">All evaluation runs</p>
                </div>
                <button class="btn btn-primary" onclick="showNewRunModal()">+ New Run</button>
            </div>
            <div class="empty-state">
                <h3>No runs yet</h3>
                <p>Run your first evaluation to see results here.</p>
                <button class="btn btn-primary" onclick="navigateTo('datasets')">Create Dataset</button>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="page-header">
            <div>
                <h1 class="page-title">Runs</h1>
                <p class="page-subtitle">All evaluation runs</p>
            </div>
            <button class="btn btn-primary" onclick="showNewRunModal()">+ New Run</button>
        </div>
        <div class="table-container">
            <div id="runs-list">
                ${runs.map(r => renderRunListItem(r)).join('')}
            </div>
        </div>
    `;
}

function renderRunListItem(r) {
    const dataset = datasets.find(d => d.id === r.dataset_id);
    const datasetName = dataset ? dataset.name : `Dataset #${r.dataset_id}`;
    const rateClass = r.score >= 80 ? 'rate-high' : (r.score >= 60 ? 'rate-mid' : 'rate-low');
    const isRunning = r.status === 'running';

    return `
        <div class="run-list-item" onclick="openRunSidebar(${r.id})" id="run-item-${r.id}">
            <div class="run-list-info">
                <div class="run-list-name">
                    ${isRunning ? '<span class="status-dot running" style="margin-right: 6px;"></span>' : ''}
                    ${escapeHtml(datasetName)}
                    ${isRunning ? ' <span style="color: var(--taro);">Running</span>' : ''}
                </div>
                <div class="run-list-sub">${relativeTime(r.started_at)} · ${r.total} cases · ${escapeHtml(r.eval_name)}</div>
            </div>
            <span class="run-list-score ${rateClass}">
                ${isRunning ? '—' : (r.score !== null ? r.score.toFixed(0) + '%' : '—')}
            </span>
            <span class="run-list-arrow">›</span>
        </div>
    `;
}

async function openRunSidebar(runId) {
    selectedRunId = runId;

    // Update selected state
    document.querySelectorAll('.run-list-item').forEach(el => el.classList.remove('selected'));
    document.getElementById(`run-item-${runId}`)?.classList.add('selected');

    // Show loading state
    document.getElementById('sidebar-run-title').textContent = 'Loading...';
    document.getElementById('sidebar-run-content').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Loading run details...
        </div>
    `;

    // Show sidebar
    document.getElementById('sidebar-overlay').classList.add('active');
    document.getElementById('run-sidebar').classList.add('active');

    // Fetch run details
    try {
        const response = await fetch(`${API_BASE}/runs/${runId}`);
        const run = await response.json();
        renderRunSidebarContent(run);
    } catch (e) {
        document.getElementById('sidebar-run-content').innerHTML = `
            <div class="empty-state" style="padding: 40px;">
                <p>Failed to load run details</p>
            </div>
        `;
    }
}

function renderRunSidebarContent(r) {
    const dataset = datasets.find(d => d.id === r.dataset_id);
    const datasetName = dataset ? dataset.name : `Dataset #${r.dataset_id}`;
    const results = r.results || [];

    const duration = r.completed_at
        ? formatDuration(new Date(r.completed_at) - new Date(r.started_at))
        : 'In progress';

    // Update title
    document.getElementById('sidebar-run-title').textContent = datasetName;

    // Update content
    const content = document.getElementById('sidebar-run-content');
    content.innerHTML = `
        <div class="sidebar-section">
            <div class="meta" style="margin-bottom: 16px;">
                <span>${relativeTime(r.started_at)}</span>
                <span>${duration}</span>
                <span>${escapeHtml(r.eval_name)}</span>
            </div>

            <div class="sidebar-stats">
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value">${r.total}</div>
                    <div class="sidebar-stat-label">Total</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value" style="color: var(--green-500);">${r.passed}</div>
                    <div class="sidebar-stat-label">Passed</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value" style="color: var(--red-500);">${r.failed}</div>
                    <div class="sidebar-stat-label">Failed</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-value" style="color: var(--taro);">${r.score !== null ? r.score.toFixed(0) + '%' : '—'}</div>
                    <div class="sidebar-stat-label">Score</div>
                </div>
            </div>
        </div>

        <div class="sidebar-section">
            <div class="sidebar-section-label">Results</div>
            ${results.map(res => renderSidebarResultItem(res)).join('')}
        </div>
    `;

    // Wire up delete button
    document.getElementById('sidebar-run-delete-btn').onclick = () => {
        if (confirm('Delete this run?')) {
            deleteRun(r.id);
        }
    };
}

function renderSidebarResultItem(res) {
    const c = res.case || {};
    const name = c.name || `Case #${res.case_id}`;

    return `
        <div class="result-list-item" id="sidebar-result-${res.id}">
            <div class="result-list-header" onclick="toggleSidebarResult(${res.id})">
                <span class="result-list-expand">›</span>
                <div class="result-list-info">
                    <span class="result-list-name">${escapeHtml(name)}</span>
                </div>
                <span class="badge ${res.passed ? 'badge-pass' : 'badge-fail'}" style="margin-left: auto;">
                    <span class="status-dot ${res.passed ? 'pass' : 'fail'}"></span>
                    ${res.passed ? 'Pass' : 'Fail'}
                </span>
            </div>
            <div class="result-list-detail">
                <div style="margin-bottom: 12px;">
                    <div class="detail-box-label">Expected</div>
                    <div style="font-size: 13px;">${escapeHtml(c.expected_outcome || '—')}</div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div class="detail-box-label">Actual</div>
                    <div style="font-size: 13px; padding: 8px; background: white; border-radius: 4px; border-left: 3px solid ${res.passed ? 'var(--green-500)' : 'var(--red-500)'};">
                        ${escapeHtml(res.actual_output || '—')}
                    </div>
                </div>
                ${res.reasoning ? `
                    <div>
                        <div class="detail-box-label">Reasoning</div>
                        <div style="font-size: 13px;">${escapeHtml(res.reasoning)}</div>
                    </div>
                ` : ''}
                ${res.error_message ? `
                    <div style="margin-top: 12px; padding: 8px; background: var(--red-50); border-radius: 4px; color: #991b1b;">
                        <div class="detail-box-label" style="color: #991b1b;">Error</div>
                        <div style="font-size: 13px;">${escapeHtml(res.error_message)}</div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

function toggleSidebarResult(id) {
    const item = document.getElementById(`sidebar-result-${id}`);
    if (item) {
        item.classList.toggle('expanded');
    }
}

function closeRunSidebar() {
    selectedRunId = null;
    document.getElementById('sidebar-overlay').classList.remove('active');
    document.getElementById('run-sidebar').classList.remove('active');
    document.querySelectorAll('.run-list-item').forEach(el => el.classList.remove('selected'));
}

async function deleteRun(id) {
    try {
        await fetch(`${API_BASE}/runs/${id}`, { method: 'DELETE' });
        closeRunSidebar();
        await loadRuns();
        renderRunsPage();
        showToast('Run deleted');
    } catch (e) {
        showToast('Failed to delete run', true);
    }
}

// Modals
function showModal(id) {
    document.getElementById(id).classList.add('active');
}

function hideModal(id) {
    document.getElementById(id).classList.remove('active');
}

// New Dataset Modal
function showNewDatasetModal() {
    document.getElementById('new-dataset-form').innerHTML = `
        <div class="form-group">
            <label for="product-description">Describe your product</label>
            <textarea id="product-description" rows="4" placeholder="A customer support chatbot for an e-commerce site that handles order inquiries, returns, and shipping questions..."></textarea>
            <p class="form-hint">The AI will generate a dataset name, description, and test cases based on your description.</p>
        </div>
    `;

    document.getElementById('new-dataset-footer').innerHTML = `
        <button class="btn btn-secondary" onclick="hideModal('modal-new-dataset')">Cancel</button>
        <button class="btn btn-primary" onclick="generateDataset()">Generate Dataset</button>
    `;

    showModal('modal-new-dataset');
}

async function generateDataset() {
    const productDesc = document.getElementById('product-description').value.trim();

    if (!productDesc) {
        showToast('Please describe your product', true);
        return;
    }

    document.getElementById('new-dataset-form').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Generating dataset...
        </div>
    `;
    document.getElementById('new-dataset-footer').innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/datasets/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_description: productDesc })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Generation failed');
        }

        const dataset = await response.json();
        hideModal('modal-new-dataset');
        await loadDatasets();
        showToast(`Created dataset "${dataset.name}" with ${dataset.case_count} cases`);
        viewDataset(dataset.id);
    } catch (e) {
        showToast(e.message || 'Failed to generate dataset', true);
        showNewDatasetModal();
    }
}

async function createDataset(name, description) {
    try {
        const response = await fetch(`${API_BASE}/datasets`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description })
        });

        if (!response.ok) throw new Error('Failed to create dataset');

        const dataset = await response.json();
        hideModal('modal-new-dataset');
        await loadDatasets();
        showToast('Dataset created');
        viewDataset(dataset.id);
    } catch (e) {
        showToast('Failed to create dataset', true);
    }
}

// Delete Dataset
function confirmDeleteDataset(id, name, caseCount) {
    document.getElementById('delete-dataset-message').innerHTML = `
        Are you sure you want to delete <strong>${escapeHtml(name)}</strong>?
        This will permanently remove the dataset and all ${caseCount} cases.
        This action cannot be undone.
    `;

    document.getElementById('confirm-delete-dataset').onclick = () => deleteDataset(id);
    showModal('modal-delete-dataset');
}

async function deleteDataset(id) {
    try {
        await fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' });
        hideModal('modal-delete-dataset');
        await loadDatasets();
        showToast('Dataset deleted');
        window.location.hash = 'datasets';
    } catch (e) {
        showToast('Failed to delete dataset', true);
    }
}

// Case Modal
let editingCaseId = null;

function showAddCaseModal() {
    editingCaseId = null;
    document.getElementById('case-modal-title').textContent = 'Add Case';
    renderCaseForm({});
    showModal('modal-case');
}

function showEditCaseModal(id) {
    editingCaseId = id;
    const c = currentDataset.cases.find(c => c.id === id);
    if (!c) return;

    document.getElementById('case-modal-title').textContent = 'Edit Case';
    renderCaseForm(c);
    showModal('modal-case');
}

function renderCaseForm(c) {
    const inputs = c.inputs || [{ role: 'user', message: '' }];

    document.getElementById('case-modal-body').innerHTML = `
        <div class="form-group">
            <label for="case-name">Name (optional)</label>
            <input type="text" id="case-name" value="${escapeHtml(c.name || '')}">
        </div>
        <div class="form-group">
            <label>Conversation</label>
            <div id="message-editor">
                ${inputs.map((m, i) => `
                    <div class="message-editor-row" style="margin-bottom: 8px; display: flex; gap: 8px;">
                        <select class="msg-role" style="width: 100px;">
                            <option value="user" ${m.role === 'user' ? 'selected' : ''}>User</option>
                            <option value="assistant" ${m.role === 'assistant' ? 'selected' : ''}>Assistant</option>
                        </select>
                        <input type="text" class="msg-content" value="${escapeHtml(m.message)}" style="flex: 1;">
                        <button class="btn btn-ghost btn-sm danger" onclick="this.parentElement.remove()">×</button>
                    </div>
                `).join('')}
            </div>
            <button class="btn btn-secondary btn-sm" onclick="addMessageRow()">+ Add Message</button>
        </div>
        <div class="form-group">
            <label for="case-outcome">Expected Outcome</label>
            <textarea id="case-outcome">${escapeHtml(c.expected_outcome || '')}</textarea>
        </div>
        ${c.expected_source ? `
            <div class="source-ref" style="margin-bottom: 16px;">
                <span class="mono">${escapeHtml(c.expected_source.file)}</span>
                <span class="source-page">p. ${c.expected_source.page}</span>
            </div>
        ` : ''}
    `;

    document.getElementById('case-modal-save').onclick = saveCase;
}

function addMessageRow() {
    const editor = document.getElementById('message-editor');
    const row = document.createElement('div');
    row.className = 'message-editor-row';
    row.style.cssText = 'margin-bottom: 8px; display: flex; gap: 8px;';
    row.innerHTML = `
        <select class="msg-role" style="width: 100px;">
            <option value="user">User</option>
            <option value="assistant">Assistant</option>
        </select>
        <input type="text" class="msg-content" style="flex: 1;">
        <button class="btn btn-ghost btn-sm danger" onclick="this.parentElement.remove()">×</button>
    `;
    editor.appendChild(row);
}

async function saveCase() {
    const name = document.getElementById('case-name').value.trim();
    const outcome = document.getElementById('case-outcome').value.trim();

    const inputs = [];
    document.querySelectorAll('.message-editor-row').forEach(row => {
        const role = row.querySelector('.msg-role').value;
        const message = row.querySelector('.msg-content').value.trim();
        if (message) {
            inputs.push({ role, message, attachments: [] });
        }
    });

    if (inputs.length === 0) {
        showToast('Please add at least one message', true);
        return;
    }

    if (!outcome) {
        showToast('Please enter an expected outcome', true);
        return;
    }

    try {
        if (editingCaseId) {
            await fetch(`${API_BASE}/cases/${editingCaseId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, inputs, expected_outcome: outcome })
            });
        } else {
            await fetch(`${API_BASE}/cases`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: currentDataset.id,
                    name,
                    inputs,
                    expected_outcome: outcome
                })
            });
        }

        hideModal('modal-case');
        await loadDatasetDetail(currentDataset.id);
        showToast(editingCaseId ? 'Case updated' : 'Case added');
    } catch (e) {
        showToast('Failed to save case', true);
    }
}

async function deleteCase(id) {
    if (!confirm('Delete this case?')) return;

    try {
        await fetch(`${API_BASE}/cases/${id}`, { method: 'DELETE' });
        await loadDatasetDetail(currentDataset.id);
        showToast('Case deleted');
    } catch (e) {
        showToast('Failed to delete case', true);
    }
}

// New Run Modal
function showNewRunModal() {
    const datasetSelect = document.getElementById('run-dataset');
    const evalSelect = document.getElementById('run-eval');

    datasetSelect.innerHTML = datasets.map(d =>
        `<option value="${d.id}">${escapeHtml(d.name)} (${d.case_count} cases)</option>`
    ).join('');

    if (evals.length === 0) {
        evalSelect.innerHTML = '<option value="">No evals loaded</option>';
        document.getElementById('run-eval-hint').innerHTML = 'Start server with --config evals.py';
    } else {
        evalSelect.innerHTML = evals.map(e =>
            `<option value="${e.name}">${escapeHtml(e.name)}</option>`
        ).join('');
        document.getElementById('run-eval-hint').innerHTML = '<a href="#" onclick="showTestModal(); return false;" style="color: var(--taro);">Test connection first →</a>';
    }

    showModal('modal-new-run');
}

// Test Connection Modal
function showTestModal() {
    hideModal('modal-new-run');

    const evalSelect = document.getElementById('test-eval');
    if (evals.length === 0) {
        evalSelect.innerHTML = '<option value="">No evals loaded</option>';
    } else {
        evalSelect.innerHTML = evals.map(e =>
            `<option value="${e.name}">${escapeHtml(e.name)}</option>`
        ).join('');
    }

    // Reset result area
    document.getElementById('test-result').style.display = 'none';
    document.getElementById('test-result').innerHTML = '';

    showModal('modal-test');
}

async function runConnectionTest() {
    const evalName = document.getElementById('test-eval').value;
    const message = document.getElementById('test-message').value.trim();
    const resultDiv = document.getElementById('test-result');

    if (!evalName) {
        showToast('No eval selected', true);
        return;
    }

    // Show loading
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <div class="loading" style="padding: 20px;">
            <div class="spinner"></div>
            Testing connection...
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/evals/${encodeURIComponent(evalName)}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const result = await response.json();

        if (result.success) {
            resultDiv.innerHTML = `
                <div style="background: var(--green-50); border: 1px solid var(--green-500); border-radius: 4px; padding: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span class="status-dot pass"></span>
                        <strong style="color: #166534;">Connection successful!</strong>
                        <span class="mono" style="color: var(--zinc-500); margin-left: auto;">${result.elapsed_ms}ms</span>
                    </div>
                    <div style="font-size: 13px; color: var(--zinc-600);">
                        <div class="detail-box-label">Response</div>
                        <div style="background: white; padding: 8px; border-radius: 4px; margin-top: 4px; max-height: 150px; overflow-y: auto;">
                            ${escapeHtml(result.response)}
                        </div>
                    </div>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div style="background: var(--red-50); border: 1px solid var(--red-500); border-radius: 4px; padding: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span class="status-dot fail"></span>
                        <strong style="color: #991b1b;">Connection failed</strong>
                        <span class="mono" style="color: var(--zinc-500); margin-left: auto;">${result.elapsed_ms}ms</span>
                    </div>
                    <div style="font-size: 13px; color: #991b1b;">
                        <div class="detail-box-label" style="color: #991b1b;">Error</div>
                        <div style="background: white; padding: 8px; border-radius: 4px; margin-top: 4px;">
                            ${escapeHtml(result.error)}
                        </div>
                    </div>
                </div>
            `;
        }
    } catch (e) {
        resultDiv.innerHTML = `
            <div style="background: var(--red-50); border: 1px solid var(--red-500); border-radius: 4px; padding: 12px; color: #991b1b;">
                <strong>Request failed:</strong> ${escapeHtml(e.message)}
            </div>
        `;
    }
}

async function startNewRun() {
    const datasetId = document.getElementById('run-dataset').value;
    const evalName = document.getElementById('run-eval').value;

    if (!datasetId || !evalName) {
        showToast('Please select a dataset and eval', true);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/runs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: parseInt(datasetId), eval_name: evalName })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start run');
        }

        const run = await response.json();
        hideModal('modal-new-run');
        await loadRuns();
        showToast('Run completed');
        openRunSidebar(run.id);
    } catch (e) {
        showToast(e.message, true);
    }
}

// File Handling
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    addFiles(files);
}

function addFiles(files) {
    for (const file of files) {
        if (file.type === 'application/pdf' && !uploadedFiles.find(f => f.name === file.name)) {
            uploadedFiles.push(file);
        }
    }
    updateFileList();
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileList();
    event.stopPropagation();
}

function updateFileList() {
    const dropText = document.getElementById('file-drop-text');
    const fileList = document.getElementById('file-list');
    if (!dropText || !fileList) return;

    if (uploadedFiles.length === 0) {
        dropText.style.display = 'block';
        fileList.style.display = 'none';
        fileList.innerHTML = '';
    } else {
        dropText.style.display = 'none';
        fileList.style.display = 'block';
        fileList.innerHTML = uploadedFiles.map((f, i) => `
            <div class="file-item">
                <span class="mono" style="font-size: 13px;">${escapeHtml(f.name)}</span>
                <button class="btn btn-ghost btn-sm" onclick="removeFile(${i})">×</button>
            </div>
        `).join('');
    }
}

function setupDragAndDrop() {
    document.addEventListener('dragover', (e) => {
        const dropZone = document.getElementById('file-drop-zone');
        if (dropZone && dropZone.contains(e.target)) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }
    });

    document.addEventListener('dragleave', (e) => {
        const dropZone = document.getElementById('file-drop-zone');
        if (dropZone && !dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('dragover');
        }
    });

    document.addEventListener('drop', (e) => {
        const dropZone = document.getElementById('file-drop-zone');
        if (dropZone && dropZone.contains(e.target)) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            addFiles(Array.from(e.dataTransfer.files));
        }
    });
}

// Settings Page
const AVAILABLE_MODELS = [
    { id: 'gemini/gemini-3-flash-preview', name: 'Gemini 3 Flash', provider: 'Google' },
    { id: 'gemini/gemini-3-pro-preview', name: 'Gemini 3 Pro', provider: 'Google' },
    { id: 'gpt-5.2', name: 'GPT-5.2', provider: 'OpenAI' },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', provider: 'Anthropic' },
    { id: 'claude-sonnet-4-5-20250929', name: 'Claude Sonnet 4.5', provider: 'Anthropic' },
    { id: 'claude-opus-4-5-20251101', name: 'Claude Opus 4.5', provider: 'Anthropic' },
];

function renderSettingsPage() {
    const container = document.getElementById('page-settings');
    const currentModel = settings.model || AVAILABLE_MODELS[0].id;

    container.innerHTML = `
        <div class="page-header">
            <h2>Settings</h2>
        </div>

        <div class="settings-section">
            <h3>LLM Configuration</h3>
            <p class="text-secondary" style="margin-bottom: 16px;">
                Configure the model used for generating test cases and judging results.
            </p>

            <div class="input-group" style="max-width: 400px;">
                <label for="settings-model">Model</label>
                <select id="settings-model">
                    ${AVAILABLE_MODELS.map(m => `
                        <option value="${m.id}" ${m.id === currentModel ? 'selected' : ''}>
                            ${m.name} (${m.provider})
                        </option>
                    `).join('')}
                </select>
            </div>

            <div style="margin-top: 24px;">
                <button class="btn btn-primary" onclick="handleSaveSettings()">Save Settings</button>
            </div>

            <div class="api-key-info" style="margin-top: 32px; padding: 16px; background: var(--zinc-100); border-radius: 4px;">
                <h4 style="margin-bottom: 8px;">API Keys</h4>
                <p class="text-secondary" style="font-size: 13px; margin: 0;">
                    Set the appropriate environment variable for your provider before starting simboba:
                </p>
                <pre style="margin: 12px 0 0; padding: 12px; background: var(--zinc-900); color: var(--zinc-100); border-radius: 4px; font-size: 12px; overflow-x: auto;">export GEMINI_API_KEY=...
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...</pre>
            </div>
        </div>
    `;
}

async function handleSaveSettings() {
    const model = document.getElementById('settings-model').value;
    const success = await saveSettings({ model });
    if (success) {
        showToast('Settings saved');
    } else {
        showToast('Failed to save settings', true);
    }
}

// Utilities
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, len) {
    if (!text) return '';
    return text.length > len ? text.slice(0, len) + '...' : text;
}

function relativeTime(dateStr) {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;

    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins} min ago`;
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return `${Math.floor(days / 7)} week${days >= 14 ? 's' : ''} ago`;
}

function formatDuration(ms) {
    const secs = Math.floor(ms / 1000);
    if (secs < 60) return `${secs}s`;
    const mins = Math.floor(secs / 60);
    return `${mins}m ${secs % 60}s`;
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast show' + (isError ? ' error' : '');

    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

// Playground Page
function renderPlaygroundPage() {
    const container = document.getElementById('playground-content');

    container.innerHTML = `
        <div class="page-header">
            <div>
                <h1 class="page-title">Playground</h1>
                <p class="page-subtitle">Query your data using natural language</p>
            </div>
        </div>

        <div class="playground-input-section" style="margin-bottom: 24px;">
            <div class="input-group" style="margin-bottom: 12px;">
                <label for="playground-query">Ask a question about your data</label>
                <div style="display: flex; gap: 8px;">
                    <input type="text" id="playground-query"
                        placeholder="e.g., Find last 10 results of runs"
                        style="flex: 1;"
                        onkeydown="if(event.key === 'Enter') runPlaygroundQuery()">
                    <button class="btn btn-primary" onclick="runPlaygroundQuery()">Run Query</button>
                </div>
            </div>
            <div class="example-queries" style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="color: var(--zinc-500); font-size: 13px;">Examples:</span>
                <button class="btn btn-ghost btn-sm" onclick="setPlaygroundQuery('Show all datasets')">Show all datasets</button>
                <button class="btn btn-ghost btn-sm" onclick="setPlaygroundQuery('Find last 10 run results')">Last 10 run results</button>
                <button class="btn btn-ghost btn-sm" onclick="setPlaygroundQuery('Show runs with score below 50%')">Runs below 50%</button>
                <button class="btn btn-ghost btn-sm" onclick="setPlaygroundQuery('Count cases per dataset')">Cases per dataset</button>
            </div>
        </div>

        <div id="playground-results"></div>
    `;
}

function setPlaygroundQuery(query) {
    document.getElementById('playground-query').value = query;
    runPlaygroundQuery();
}

async function runPlaygroundQuery() {
    const query = document.getElementById('playground-query').value.trim();
    const resultsDiv = document.getElementById('playground-results');

    if (!query) {
        showToast('Please enter a query', true);
        return;
    }

    // Show loading
    resultsDiv.innerHTML = `
        <div class="loading" style="padding: 40px;">
            <div class="spinner"></div>
            Generating and executing query...
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/playground/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const result = await response.json();

        if (result.success) {
            renderPlaygroundResults(result);
        } else {
            resultsDiv.innerHTML = `
                <div style="background: var(--red-50); border: 1px solid var(--red-500); border-radius: 4px; padding: 16px;">
                    <div style="color: #991b1b; font-weight: 500; margin-bottom: 8px;">Query failed</div>
                    <div style="font-size: 13px; color: #991b1b;">${escapeHtml(result.error)}</div>
                    ${result.sql ? `
                        <div style="margin-top: 12px;">
                            <div class="detail-box-label">Generated SQL</div>
                            <pre class="mono" style="background: white; padding: 8px; border-radius: 4px; font-size: 12px; overflow-x: auto;">${escapeHtml(result.sql)}</pre>
                        </div>
                    ` : ''}
                </div>
            `;
        }
    } catch (e) {
        resultsDiv.innerHTML = `
            <div style="background: var(--red-50); border: 1px solid var(--red-500); border-radius: 4px; padding: 16px; color: #991b1b;">
                <strong>Request failed:</strong> ${escapeHtml(e.message)}
            </div>
        `;
    }
}

function renderPlaygroundResults(result) {
    const resultsDiv = document.getElementById('playground-results');
    const { sql, columns, results } = result;

    if (results.length === 0) {
        resultsDiv.innerHTML = `
            <div style="margin-bottom: 16px;">
                <div class="detail-box-label">Generated SQL</div>
                <pre class="mono" style="background: var(--zinc-100); padding: 12px; border-radius: 4px; font-size: 12px; overflow-x: auto;">${escapeHtml(sql)}</pre>
            </div>
            <div class="empty-state" style="padding: 40px;">
                <h3>No results</h3>
                <p>The query returned no rows.</p>
            </div>
        `;
        return;
    }

    // Build table
    const tableHtml = `
        <div style="margin-bottom: 16px;">
            <div class="detail-box-label">Generated SQL</div>
            <pre class="mono" style="background: var(--zinc-100); padding: 12px; border-radius: 4px; font-size: 12px; overflow-x: auto;">${escapeHtml(sql)}</pre>
        </div>
        <div style="margin-bottom: 8px; color: var(--zinc-500); font-size: 13px;">
            ${results.length} row${results.length !== 1 ? 's' : ''} returned
        </div>
        <div class="table-container" style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        ${columns.map(col => `<th>${escapeHtml(col)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${results.map(row => `
                        <tr>
                            ${columns.map(col => `<td class="mono" style="font-size: 12px;">${formatCellValue(row[col])}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;

    resultsDiv.innerHTML = tableHtml;
}

function formatCellValue(value) {
    if (value === null || value === undefined) {
        return '<span style="color: var(--zinc-400);">null</span>';
    }
    if (typeof value === 'object') {
        return escapeHtml(JSON.stringify(value));
    }
    if (typeof value === 'boolean') {
        return value ? '<span style="color: var(--green-500);">true</span>' : '<span style="color: var(--red-500);">false</span>';
    }
    // Truncate long strings
    const str = String(value);
    if (str.length > 100) {
        return escapeHtml(str.substring(0, 100) + '...');
    }
    return escapeHtml(str);
}

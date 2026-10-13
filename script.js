let uploadedFile = null;
let allStudents = [];
let currentFilter = 'all';

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    
    // Upload handling
    const uploadButton = document.getElementById('upload-button');
    const analyzeButton = document.getElementById('analyze-button');
    const uploadStatus = document.getElementById('upload-status');

    if (uploadButton) {
        uploadButton.addEventListener('change', (e) => {
            console.log('File selected');
            if (e.target.files.length > 0) {
                uploadedFile = e.target.files[0];
                uploadStatus.textContent = `Selected file: ${uploadedFile.name}`;
                uploadStatus.style.display = 'block';
                analyzeButton.disabled = false;
            } else {
                uploadedFile = null;
                uploadStatus.style.display = 'none';
                analyzeButton.disabled = true;
            }
        });
    }

    if (analyzeButton) {
        analyzeButton.addEventListener('click', async () => {
            console.log('Analyze button clicked');
            await handleAnalyze();
        });
    }

    // Filter event listeners
    const filterAll = document.getElementById('filter-all');
    const filterLow = document.getElementById('filter-low');
    const filterMedium = document.getElementById('filter-medium');
    const filterHigh = document.getElementById('filter-high');

    if (filterAll) filterAll.addEventListener('click', () => filterStudents('all'));
    if (filterLow) filterLow.addEventListener('click', () => filterStudents('low'));
    if (filterMedium) filterMedium.addEventListener('click', () => filterStudents('medium'));
    if (filterHigh) filterHigh.addEventListener('click', () => filterStudents('high'));

    // Add Student form
    const addStudentForm = document.getElementById('add-student-form');
    if (addStudentForm) addStudentForm.addEventListener('submit', handleAddStudent);
    const resetDemoButton = document.getElementById('reset-demo-button');
    if (resetDemoButton) resetDemoButton.addEventListener('click', handleResetDemo);
    const exportButton = document.getElementById('export-button');
    if (exportButton) exportButton.addEventListener('click', handleExport);

    // Initialize dashboard
    fetchFeatures();
    fetchStudents();
});

async function handleAnalyze() {
    if (!uploadedFile) {
        alert('Please select a file to upload first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', uploadedFile);

    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.textContent = 'Uploading and analyzing...';

    try {
        console.log('Starting upload...');
        const response = await fetch('http://127.0.0.1:5050/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Upload response:', result);

        if (!response.ok) {
            alert(`Error: ${result.error || 'Unknown error'}`);
            uploadStatus.textContent = '';
            return;
        }

        uploadStatus.textContent = result.message;
        document.getElementById('analyze-button').disabled = true;
        document.getElementById('upload-button').value = '';
        uploadedFile = null;

        // Refresh data
        await fetchFeatures();
        await fetchStudents();

    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed: ' + error.message);
        uploadStatus.textContent = '';
    }
}

async function fetchFeatures() {
    try {
        console.log('Fetching features...');
        const response = await fetch('http://127.0.0.1:5050/api/features');
        if (!response.ok) throw new Error('Failed fetching features');
        const features = await response.json();
        renderFeatures(features);
        console.log('Features rendered');
    } catch (e) {
        console.error('Features error:', e);
        const featureList = document.getElementById('feature-list');
        if (featureList) {
            featureList.textContent = 'Error loading features';
        }
    }
}

function renderFeatures(features) {
    const container = document.getElementById('feature-list');
    if (!container) return;
    
    container.innerHTML = '';
    features.forEach(feature => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.innerHTML = `
            <span>${feature.name}</span>
            <span>${feature.importance}%</span>
        `;
        container.appendChild(item);
    });
}

async function fetchStudents() {
    try {
        console.log('Fetching students...');
        const response = await fetch('http://127.0.0.1:5050/api/students');
        if (!response.ok) throw new Error('Cannot load students');
        const students = await response.json();
        console.log('Students fetched:', students.length);
        renderStudents(students);
        clearExplanation();
    } catch (e) {
        console.error('Students error:', e);
        const tbody = document.getElementById('student-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="3">Upload a file or add a student to get started</td></tr>';
        }
        allStudents = [];
        updateSummaryCards([]);
    }
}

function renderStudents(students) {
    console.log('Rendering students:', students.length);
    // Students added through the form go first, newest on top
    const added = students.filter(s => s.manual).sort((a, b) => b.id - a.id);
    allStudents = [...added, ...students.filter(s => !s.manual)];
    students = allStudents;
    displayStudents(students);
    updateFilterStats();
    updateSummaryCards(students);
}

function updateSummaryCards(students) {
    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    if (!students.length) {
        ['stat-total', 'stat-avg-risk', 'stat-moderate', 'stat-attention'].forEach(id => setText(id, '—'));
        return;
    }
    const avgRisk = students.reduce((sum, s) => sum + (Number(s.dropout_risk) || 0), 0) / students.length;
    setText('stat-total', students.length);
    setText('stat-avg-risk', `${avgRisk.toFixed(1)}%`);
    setText('stat-moderate', students.filter(s => s.riskClass === 'medium').length);
    setText('stat-attention', students.filter(s => s.riskClass === 'high').length);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setFormStatus(message, type) {
    const status = document.getElementById('add-student-status');
    if (!status) return;
    status.textContent = message;
    status.className = `form-status ${type || ''}`;
}

async function handleAddStudent(event) {
    event.preventDefault();
    const field = id => document.getElementById(id);
    const nameInput = field('student-name');
    const name = nameInput.value.trim();
    if (!name) return setFormStatus('Please enter a student name.', 'error');

    // input id -> [payload key, label, min, max]
    const numericFields = [
        ['student-age', 'age', 'Age', 15, 22],
        ['student-failures', 'failures', 'Past failures', 0, 4],
        ['student-absences', 'absences', 'Absences', 0, 93],
    ];
    const payload = { name };
    for (const [id, key, label, min, max] of numericFields) {
        const raw = field(id).value.trim();
        const value = Number(raw);
        if (raw === '' || !Number.isInteger(value) || value < min || value > max) {
            return setFormStatus(`${label} must be a whole number from ${min} to ${max}.`, 'error');
        }
        payload[key] = value;
    }
    payload.studytime = Number(field('student-studytime').value);
    payload.higher = field('student-higher').value;
    payload.famsup = field('student-famsup').value;
    payload.internet = field('student-internet').value;

    const button = document.getElementById('add-student-button');
    button.disabled = true;
    setFormStatus('Adding student...');

    try {
        const response = await fetch('http://127.0.0.1:5050/api/students', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);

        setFormStatus(result.message, 'success');
        document.getElementById('add-student-form').reset();
        nameInput.focus();

        await fetchFeatures();
        await fetchStudents();
        filterStudents('all');
        const newRow = document.querySelector(`#student-tbody tr[data-student-id="${result.id}"]`);
        if (newRow) newRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (e) {
        console.error('Add student error:', e);
        const reason = e instanceof TypeError
            ? 'the backend is not running. Start it with: .\\.venv\\Scripts\\python.exe app.py'
            : e.message;
        setFormStatus(`Could not add student: ${reason}`, 'error');
    } finally {
        button.disabled = false;
    }
}

async function handleExport() {
    const button = document.getElementById('export-button');
    button.disabled = true;
    try {
        const response = await fetch('http://127.0.0.1:5050/api/export');
        if (!response.ok) {
            const result = await response.json().catch(() => ({}));
            throw new Error(result.error || `HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `student_dropout_risk_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(link.href);
    } catch (e) {
        console.error('Export error:', e);
        const reason = e instanceof TypeError ? 'the backend is not running' : e.message;
        alert(`Export failed: ${reason}`);
    } finally {
        button.disabled = false;
    }
}

async function handleResetDemo() {
    const button = document.getElementById('reset-demo-button');
    button.disabled = true;
    setFormStatus('Reloading demo data...');
    try {
        const response = await fetch('http://127.0.0.1:5050/api/reset_demo', { method: 'POST' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
        setFormStatus(`${result.message} (${result.num_students} students)`, 'success');
        await fetchFeatures();
        await fetchStudents();
        filterStudents('all');
    } catch (e) {
        console.error('Reset demo error:', e);
        setFormStatus(`Could not reset demo: ${e.message}`, 'error');
    } finally {
        button.disabled = false;
    }
}

function displayStudents(students) {
    const tbody = document.getElementById('student-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    students.forEach(student => {
        const row = document.createElement('tr');
        row.dataset.studentId = student.id;
        row.className = student.manual ? 'student-row added' : 'student-row';

        row.innerHTML = `
            <td>${escapeHtml(student.name)} (${student.id})</td>
            <td><span class="risk-${student.riskClass}">${student.dropout_risk}%</span></td>
            <td><button class="view-details-btn" data-student-id="${student.id}">View Details</button></td>
        `;

        // Add event listeners for both row click and button click
        row.addEventListener('click', (e) => {
            // Don't trigger if button was clicked
            if (e.target.tagName !== 'BUTTON') {
                selectStudent(student.id);
            }
        });

        // Add specific button click handler
        const button = row.querySelector('.view-details-btn');
        if (button) {
            button.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent row click
                console.log('View details button clicked for student:', student.id);
                selectStudent(student.id);
            });
        }

        tbody.appendChild(row);
    });

    console.log('Students displayed successfully');
}

function updateFilterStats() {
    const total = allStudents.length;
    const low = allStudents.filter(s => s.riskClass === 'low').length;
    const medium = allStudents.filter(s => s.riskClass === 'medium').length;
    const high = allStudents.filter(s => s.riskClass === 'high').length;
    
    const filterStats = document.getElementById('filter-stats');
    if (filterStats) {
        filterStats.innerHTML = `Total: ${total} | Low Risk: ${low} | Moderate: ${medium} | High Risk: ${high}`;
    }
}

function filterStudents(filterType) {
    console.log('Filtering students:', filterType);
    currentFilter = filterType;
    
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`filter-${filterType}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Filter students
    let filteredStudents = allStudents;
    if (filterType !== 'all') {
        filteredStudents = allStudents.filter(student => student.riskClass === filterType);
    }
    
    displayStudents(filteredStudents);
    
    // Update stats
    const filterStats = document.getElementById('filter-stats');
    if (filterStats) {
        filterStats.innerHTML = `Showing ${filteredStudents.length} of ${allStudents.length} students`;
    }
}

// Student explanation - FIXED VERSION
async function selectStudent(studentId) {
    console.log(`Selecting student ${studentId} for explanation`);
    
    const explanationContent = document.getElementById('explanation-content');
    if (!explanationContent) {
        console.error('Explanation content element not found');
        return;
    }

    // Show loading
    explanationContent.innerHTML = '<p>Loading explanation...</p>';
    
    try {
        console.log('Fetching explanation...');
        const response = await fetch(`http://127.0.0.1:5050/api/explain_student?student_id=${studentId}`);
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const explanation = await response.json();
        console.log('Explanation received:', explanation.student_id, explanation.predicted_dropout_risk_percent);

        if (explanation.error) {
            explanationContent.innerHTML = `<p style="color: red;">Error: ${explanation.error}</p>`;
            return;
        }

        // Display the explanation
        let html = `
            <h3>👩‍🎓 ${explanation.counselor_name || 'Dr. Sarah Chen'}</h3>
            <p><strong>${escapeHtml(explanation.student_name || `Student ${explanation.student_id + 1}`)} - Dropout Risk: ${explanation.predicted_dropout_risk_percent}%</strong></p>
            
            <h4>📋 Institutional Assessment</h4>
            <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin: 10px 0;">
                <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">${explanation.ai_explanation}</pre>
            </div>
            
            <details style="margin-top: 15px;">
                <summary><strong>🔍 Technical Analysis Details</strong></summary>
                <ul style="margin-top: 10px; padding-left: 20px;">`;
        
        if (explanation.technical_factors && explanation.technical_factors.length > 0) {
            explanation.technical_factors.forEach(feat => {
                html += `<li style="margin-bottom: 5px;"><b>${feat.feature}</b> = ${feat.feature_value} : ${feat.interpretation} (SHAP=${feat.shap_value.toFixed(3)})</li>`;
            });
        } else {
            html += '<li>No technical factors available</li>';
        }
        
        html += `</ul></details>`;

        explanationContent.innerHTML = html;

        // Highlight selected row
        document.querySelectorAll('.student-row').forEach(r => r.classList.remove('selected'));
        const selectedRow = document.querySelector(`[data-student-id="${studentId}"]`);
        if (selectedRow) {
            selectedRow.classList.add('selected');
        }

        // Scroll to explanation
        const explanationSection = document.getElementById('explanation-section');
        if (explanationSection) {
            explanationSection.scrollIntoView({behavior: 'smooth'});
        }

        console.log('Explanation displayed successfully');

    } catch (e) {
        console.error('Student explanation error:', e);
        explanationContent.innerHTML = `<p style="color: red;">Error loading explanation: ${e.message}</p>`;
    }
}

function clearExplanation() {
    const explanationContent = document.getElementById('explanation-content');
    if (explanationContent) {
        explanationContent.innerHTML = '<p>Select a student from the table to view detailed assessment.</p>';
    }
}

// Global function for onclick handlers in HTML
window.selectStudent = selectStudent;

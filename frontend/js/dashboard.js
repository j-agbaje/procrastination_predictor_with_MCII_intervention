const API_BASE = 'http://localhost:8000';

async function loadStudentDashboard() {
    const token = localStorage.getItem('token');
    const userId = localStorage.getItem('user_id');
    
    if (!token) {
        window.location.href = 'login.html';
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                student_id: parseInt(userId),
                behavioral_data: {
                    late_rate: 0.3,
                    irregularity: 0.5,
                    last_min_ratio: 0.4,
                    avg_gap: 2.0
                }
            })
        });

        if (response.ok) {
            const prediction = await response.json();
            console.log('Risk:', prediction.risk_level, prediction.risk_score);
        }
    } catch (error) {
        console.error('Prediction failed:', error);
    }
}

function logout() {
    localStorage.clear();
    window.location.href = 'login.html';
}

window.addEventListener('DOMContentLoaded', () => {
    if (window.location.pathname.includes('student_dashboard')) {
        loadStudentDashboard();
    }
});
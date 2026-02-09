-- Students Table
CREATE TABLE Students (
    student_id      INT AUTO_INCREMENT PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    enrollment_date DATE NOT NULL,
    current_risk_level ENUM('low', 'medium', 'high') DEFAULT 'low',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Admins Table
CREATE TABLE Admins (
    admin_id        INT AUTO_INCREMENT PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    department      VARCHAR(100) NULL,
    access_level    INT DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Surveys Table
CREATE TABLE Surveys (
    survey_id       INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT UNIQUE NOT NULL,
    responses_json  JSON NOT NULL,
    completion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE
);


-- Tasks Table
CREATE TABLE Tasks (
    task_id         INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT NOT NULL,
    title           VARCHAR(200) NOT NULL,
    description     TEXT NULL,
    due_date        DATETIME NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP NULL,
    status          ENUM('pending', 'in_progress', 'completed', 'overdue') DEFAULT 'pending',
    
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
    
    INDEX idx_tasks_student_status (student_id, status)
);


-- BehavioralLogs Table
CREATE TABLE BehavioralLogs (
    log_id          INT AUTO_INCREMENT PRIMARY KEY,
    student_id      INT NOT NULL,
    login_time      TIMESTAMP NOT NULL,
    logout_time     TIMESTAMP NULL,
    pages_visited   INT DEFAULT 0,
    session_duration INT NULL,          -- in seconds
    
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
    
    INDEX idx_logs_student_time (student_id, login_time)
);


-- Predictions Table
CREATE TABLE Predictions (
    prediction_id     INT AUTO_INCREMENT PRIMARY KEY,
    student_id        INT NOT NULL,
    prediction_date   DATE NOT NULL,
    risk_level        ENUM('low', 'medium', 'high') NOT NULL,
    confidence_score  DECIMAL(3,2) NOT NULL,          -- 0.00 to 1.00
    attention_weights_json JSON NULL,
    
    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
    
    UNIQUE KEY uk_prediction_student_date (student_id, prediction_date),
    INDEX idx_predictions_student_date (student_id, prediction_date)
);


-- MCIIInterventions Table
CREATE TABLE MCIIInterventions (
    intervention_id   INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id     INT NOT NULL,
    student_id        INT NOT NULL,
    prompt_text       TEXT NOT NULL,
    delivery_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_response     TEXT NULL,
    was_helpful       BOOLEAN NULL,
    
    FOREIGN KEY (prediction_id) REFERENCES Predictions(prediction_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id)    REFERENCES Students(student_id) ON DELETE CASCADE,
    
    INDEX idx_mcii_student_time (student_id, delivery_time)
);
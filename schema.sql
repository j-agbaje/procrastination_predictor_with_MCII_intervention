-- MySQL dump 10.13  Distrib 9.3.0, for macos14.7 (x86_64)
--
-- Host: localhost    Database: proactivedb
-- ------------------------------------------------------
-- Server version	9.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Admins`
--

DROP TABLE IF EXISTS `Admins`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Admins` (
  `admin_id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `department` varchar(100) DEFAULT NULL,
  `access_level` int DEFAULT '1',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `invite_code` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`admin_id`),
  UNIQUE KEY `email` (`email`),
  UNIQUE KEY `invite_code` (`invite_code`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `BehavioralLogs`
--

DROP TABLE IF EXISTS `BehavioralLogs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `BehavioralLogs` (
  `log_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `login_time` timestamp NOT NULL,
  `logout_time` timestamp NULL DEFAULT NULL,
  `pages_visited` int DEFAULT '0',
  `session_duration` int DEFAULT NULL,
  PRIMARY KEY (`log_id`),
  KEY `idx_logs_student_time` (`student_id`,`login_time`),
  CONSTRAINT `behaviorallogs_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `MCIIInterventions`
--

DROP TABLE IF EXISTS `MCIIInterventions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `MCIIInterventions` (
  `intervention_id` int NOT NULL AUTO_INCREMENT,
  `prediction_id` int DEFAULT NULL,
  `student_id` int NOT NULL,
  `prompt_text` text NOT NULL,
  `delivery_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `user_response` text,
  `was_helpful` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`intervention_id`),
  KEY `prediction_id` (`prediction_id`),
  KEY `idx_mcii_student_time` (`student_id`,`delivery_time`),
  CONSTRAINT `mciiinterventions_ibfk_1` FOREIGN KEY (`prediction_id`) REFERENCES `Predictions` (`prediction_id`) ON DELETE CASCADE,
  CONSTRAINT `mciiinterventions_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=49 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Predictions`
--

DROP TABLE IF EXISTS `Predictions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Predictions` (
  `prediction_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `bundle_id` int DEFAULT NULL,
  `prediction_date` date NOT NULL,
  `model_used` enum('baseline','3window','7window') NOT NULL,
  `risk_level` enum('low','medium','high') NOT NULL,
  `confidence_score` decimal(3,2) NOT NULL,
  `attention_weights_json` json DEFAULT NULL,
  `features_json` json DEFAULT NULL,
  PRIMARY KEY (`prediction_id`),
  KEY `bundle_id` (`bundle_id`),
  KEY `idx_predictions_student_date` (`student_id`,`prediction_date`),
  CONSTRAINT `predictions_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE,
  CONSTRAINT `predictions_ibfk_2` FOREIGN KEY (`bundle_id`) REFERENCES `WeeklyBundles` (`bundle_id`) ON DELETE SET NULL
) ENGINE=InnoDB AUTO_INCREMENT=62 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Students`
--

DROP TABLE IF EXISTS `Students`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Students` (
  `student_id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL,
  `full_name` varchar(100) DEFAULT NULL,
  `password_hash` varchar(255) NOT NULL,
  `enrollment_date` date NOT NULL,
  `current_risk_level` enum('low','medium','high') DEFAULT 'low',
  `prior_profile` enum('early','mixed','lastminute') DEFAULT 'mixed',
  `days_active` int DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `phone` varchar(20) DEFAULT NULL,
  `profile_pic` varchar(255) DEFAULT NULL,
  `bio` text,
  `admin_id` int DEFAULT NULL,
  PRIMARY KEY (`student_id`),
  UNIQUE KEY `email` (`email`),
  KEY `fk_student_admin` (`admin_id`),
  CONSTRAINT `fk_student_admin` FOREIGN KEY (`admin_id`) REFERENCES `Admins` (`admin_id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Surveys`
--

DROP TABLE IF EXISTS `Surveys`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Surveys` (
  `survey_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `responses_json` json NOT NULL,
  `completion_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`survey_id`),
  UNIQUE KEY `student_id` (`student_id`),
  CONSTRAINT `surveys_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Tasks`
--

DROP TABLE IF EXISTS `Tasks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Tasks` (
  `task_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `bundle_id` int DEFAULT NULL,
  `title` varchar(200) NOT NULL,
  `description` text,
  `due_date` datetime DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `completed_at` timestamp NULL DEFAULT NULL,
  `status` enum('pending','in_progress','completed','overdue') DEFAULT 'pending',
  `created_by_admin_id` int DEFAULT NULL,
  `task_type` varchar(20) DEFAULT 'personal',
  `is_admin_assigned` tinyint(1) DEFAULT '0',
  PRIMARY KEY (`task_id`),
  KEY `idx_tasks_student_status` (`student_id`,`status`),
  KEY `idx_tasks_bundle` (`bundle_id`),
  CONSTRAINT `fk_tasks_bundle` FOREIGN KEY (`bundle_id`) REFERENCES `WeeklyBundles` (`bundle_id`) ON DELETE SET NULL,
  CONSTRAINT `tasks_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `WeeklyBundles`
--

DROP TABLE IF EXISTS `WeeklyBundles`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `WeeklyBundles` (
  `bundle_id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `week_number` int NOT NULL,
  `start_date` date NOT NULL,
  `end_date` date NOT NULL,
  `tasks_total` int DEFAULT '0',
  `tasks_completed` int DEFAULT '0',
  `tasks_late` int DEFAULT '0',
  `completion_rate` decimal(4,3) DEFAULT '0.000',
  `submitted_late` tinyint(1) DEFAULT '0',
  `is_closed` tinyint(1) DEFAULT '0',
  `closed_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`bundle_id`),
  UNIQUE KEY `uk_bundle_student_week` (`student_id`,`week_number`),
  KEY `idx_bundle_student_closed` (`student_id`,`is_closed`),
  KEY `idx_bundle_end_date` (`end_date`),
  CONSTRAINT `weeklybundles_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `Students` (`student_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-03-15 12:58:02

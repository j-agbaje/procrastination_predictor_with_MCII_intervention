-- MySQL dump 10.13  Distrib 9.3.0, for macos14.7 (x86_64)
--
-- Host: localhost    Database: ProActive_db
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
-- Dumping data for table `Admins`
--

LOCK TABLES `Admins` WRITE;
/*!40000 ALTER TABLE `Admins` DISABLE KEYS */;
INSERT INTO `Admins` VALUES (1,'jerry@test.com','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','Computer Science',1,'2026-02-26 16:31:34','e47c7979'),(2,'jerry@test2.com','ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f','General',1,'2026-03-13 21:40:48','8RJMPJ4C');
/*!40000 ALTER TABLE `Admins` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `BehavioralLogs`
--

LOCK TABLES `BehavioralLogs` WRITE;
/*!40000 ALTER TABLE `BehavioralLogs` DISABLE KEYS */;
INSERT INTO `BehavioralLogs` VALUES (1,1,'2026-03-01 19:49:43','2026-03-01 20:36:41',0,2817),(2,1,'2026-03-01 21:50:10','2026-03-02 00:39:25',0,10154),(3,1,'2026-03-02 01:08:15',NULL,0,NULL),(4,1,'2026-03-05 13:08:12','2026-03-05 13:08:33',0,21),(5,1,'2026-03-05 13:09:22','2026-03-05 15:05:34',0,6972),(6,1,'2026-03-05 15:29:29','2026-03-05 15:37:14',0,465),(7,1,'2026-03-05 15:37:21','2026-03-05 17:08:14',0,5452),(8,1,'2026-03-05 17:08:41','2026-03-05 17:12:45',0,243),(9,1,'2026-03-05 17:16:36','2026-03-10 16:46:24',0,430187),(10,1,'2026-03-08 23:17:55','2026-03-08 23:40:31',0,1355),(11,1,'2026-03-08 23:53:40','2026-03-09 00:12:19',0,1119),(12,1,'2026-03-09 13:25:16','2026-03-10 15:32:06',0,94010),(13,1,'2026-03-10 19:17:32','2026-03-11 16:27:26',0,76194),(14,1,'2026-03-11 16:32:05','2026-03-11 17:03:21',0,1875),(15,1,'2026-03-11 17:03:44',NULL,0,NULL),(16,1,'2026-03-12 20:40:07','2026-03-13 22:15:56',0,92149),(17,1,'2026-03-13 23:28:38',NULL,0,NULL),(18,1,'2026-03-13 23:56:52','2026-03-14 00:11:54',0,901),(19,4,'2026-03-14 00:12:14','2026-03-14 00:14:36',0,142),(20,6,'2026-03-14 00:14:51','2026-03-14 00:17:03',0,132),(21,4,'2026-03-14 00:17:17',NULL,0,NULL),(22,1,'2026-03-14 14:15:41','2026-03-14 15:04:08',0,2907),(23,4,'2026-03-14 15:05:36','2026-03-14 15:19:15',0,818),(24,1,'2026-03-14 20:33:58','2026-03-14 20:41:25',0,446),(25,1,'2026-03-14 21:22:16','2026-03-14 21:23:41',0,84);
/*!40000 ALTER TABLE `BehavioralLogs` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `MCIIInterventions`
--

LOCK TABLES `MCIIInterventions` WRITE;
/*!40000 ALTER TABLE `MCIIInterventions` DISABLE KEYS */;
INSERT INTO `MCIIInterventions` VALUES (2,4,3,'# MCII Tip: Break the Procrastination Cycle Tonight\n\n**Your Reality:** With your deadline approaching, you likely have 3-4 major tasks still pending. Last-minute work risks quality and stress.\n\n**Your Mental Contrast:**\n- **Positive outcome:** Completing tasks early means better work, sleep, and confidence.\n- **Obstacle:** The urge to delay when work feels overwhelming.\n\n**Your Implementation Intention:**\n**IF** you feel the urge to procrastinate tonight, **THEN** immediately break one task into a 15-minute micro-chunk and start it without negotiation.\n\nThis removes decision-making friction. Small momentum builds—you','2026-03-08 22:12:54',NULL,NULL),(4,37,3,'# MCII Tip: Beat Procrastination This Week\n\n**The Reality:** Time\'s almost gone this week—multiple tasks remain undone.\n\n**Mental Contrast:**\n- **Positive outcome:** Complete one high-impact task today\n- **Obstacle:** Your urge to delay\n\n**Implementation Intention:**\nPick ONE task from your bundle RIGHT NOW.\n\nCreate your if-then plan:\n*\"IF [specific trigger: time of day/location], THEN I will [concrete action: open document, write first paragraph, etc.]\"*\n\n**Example:**\n\"IF it\'s 2pm, THEN I\'ll open my essay and write the introduction.\"\n\n**Your move (','2026-03-10 19:12:59',NULL,NULL),(40,52,1,'I am struggling to stay focused','2026-03-13 00:07:26','I hear you. What\'s pulling your attention right now — is it external stuff (distractions, environment), internal stuff (racing thoughts, anxiety), or just general fatigue? And do you have something specific you\'re trying to focus *on*, or is it more of an all-day thing?\n\nThe reason I\'m asking is that \"staying focused\" can mean different things depending on what\'s actually getting in the way. Once I know what we\'re working with, I can give you something actually useful instead of generic advice.',NULL),(41,52,1,'what was the last thing i sent again','2026-03-13 00:08:36','You just asked me what you last sent — that message is it. You\'re asking me to repeat it back, which means you might be testing me or genuinely spacing out.\n\nEither way, here\'s what happened in our chat: you said you\'re struggling to stay focused, I asked what\'s pulling your attention (external distractions, internal stuff like anxiety, fatigue) and whether you have something specific you need to focus on.\n\nSo — what\'s going on? Real answer this time.',NULL),(42,52,1,'Im watching too much tv','2026-03-13 00:09:29','Okay, that\'s honest. And it makes sense — TV is basically designed to hijack your attention. Low effort, high reward. Your brain loves it.\n\nHere\'s the direct question: **Are you watching TV *instead* of doing something you need to do, or is it more that you sit down to study and then end up switching to TV partway through?**\n\nThe reason matters because the fix is different. If it\'s the second one (starting strong, then drifting), we\'re dealing with a willpower problem in the moment. If it\'s the first one (just choosing TV over work), that\'s more about priorities or avoidance.\n\nWhich one is it?',NULL),(43,59,6,'What is MCII and how does it work?','2026-03-13 23:47:42','Good question — I\'m glad you asked directly.\n\n**MCII stands for Mental Contrasting with Implementation Intentions.** It\'s a research-backed technique that works in two parts:\n\n1. **Mental Contrasting**: You vividly imagine yourself succeeding at something (your goal), then you identify the *real* obstacle standing in your way. Not vague anxiety — the actual, specific thing that\'ll try to derail you.\n\n2. **Implementation Intention**: You create an if-then plan. \"If [obstacle happens], then I [do this specific thing].\" It\'s concrete and automatic — removes the need to willpower your way through.\n\n**Why it works**: Your brain gets two things it\'s craving — a clear vision of success AND a pre-made response plan when things get hard. No last-minute scrambling, no \"I\'ll figure it out when I get there.\"\n\n**In practice with you**: Say you\'ve got a deadline coming up and a history of leaving things to the last minute. We\'d talk through what success actually looks like for you, what usually trips you up (procrastination, distraction, overwhelm?), and then build a specific if-then plan that catches you before you slip into that pattern.\n\nIt\'s not magical, but it\'s effective — and it\'s a lot more flexible than waiting until you\'re in crisis mode.\n\nDoes that make sense? Is there a specific goal or deadline you\'re thinking about right now?',NULL),(44,59,6,'I am struggling to stay focused','2026-03-13 23:48:02','That\'s real, and it matters. Before I throw solutions at you, let me understand better:\n\n**What does \"struggling to stay focused\" look like for you right now?**\n\n- Is it hard to *start* a task, or hard to *keep going* once you\'ve started?\n- Are you getting distracted by specific things (phone, other tasks, random thoughts)?\n- Or does it feel more like mental fog — like you\'re trying but nothing\'s sticking?\n- When did this become noticeable?\n\nAlso — you mentioned you\'ve got 1 day until your weekly deadline and no tasks left. Are you caught up, or is that deadline creating some pressure that\'s making focus even harder?\n\nGive me the real picture, and we\'ll figure out what\'s actually going on.',NULL),(45,57,4,'Help me make a study plan for today','2026-03-14 00:19:06','Good timing — you\'ve got 1 day until your weekly deadline and nothing logged yet, so let\'s make this count.\n\nBefore I build the plan, I need to know:\n\n1. **What\'s actually due?** (specific assignments, subjects, or exams?)\n2. **How much time do you realistically have today?**\n3. **What usually trips you up** — is it starting, staying focused, or finishing strong?\n\nI\'m asking because your history shows you\'re a last-minute person, which means a generic \"study for 3 hours\" plan won\'t stick. We need something that *actually* fits how you work and sets you up to follow through.\n\nOnce I know those three things, I\'ll give you a concrete breakdown — and more importantly, we\'ll build in some if-then triggers so you actually *do* the thing instead of planning to do it and then getting stuck.\n\nWhat\'s on the docket?',NULL),(46,57,4,'physics assignment due, i have six hours from 12pm to 6 pm and yes my main problem is staying focused i tend to daydream alot','2026-03-14 00:20:19','Okay, solid window. Six hours for physics is workable — and daydreaming is actually the *good* news here because it\'s fixable with the right structure.\n\n**Here\'s your plan:**\n\n**12:00–12:30pm** — Read the assignment brief + gather all materials (notes, textbook, examples). No work yet, just orient yourself.\n\n**12:30–1:30pm** — Tackle the first problem/section. One focused block, phone in another room.\n\n**1:30–1:45pm** — Break. Actually move around, don\'t scroll.\n\n**1:45–3:15pm** — Work block 2 (90 min). Harder stuff or bulk of the assignment.\n\n**3:15–3:30pm** — Break.\n\n**3:30–5:00pm** — Work block 3 (90 min). Finish strong.\n\n**5:00–6:00pm** — Review, polish, submit.\n\n**Why this works for you:** 90-min blocks match your focus capacity. Short breaks stop the mental drift before it kills momentum.\n\nNow — **the real part that makes this *stick*:**\n\n**If** you feel the daydream creeping in during a work block, **then** you immediately write down what you were thinking (takes 10 seconds) and jump to a different problem or section for 5 min before returning.\n\nDoes that resonate? Any part of this feel unrealistic, or do you want to lock this in?',NULL),(47,56,1,'I keep procrastinating on my work','2026-03-14 21:23:18','Yeah, TV is usually the symptom, not the root. So you\'re avoiding the work itself.\n\nBefore I throw a study plan at you: **What work are you procrastinating on?** And more importantly — what\'s making you avoid it? Is it boring, hard, unclear what to do, feeling overwhelmed, or something else?\n\nI\'m asking because if I just tell you to \"sit down and work,\" you\'ll be back on Netflix in 20 minutes. We need to figure out what\'s actually in the way first.',NULL),(48,56,1,'lllllllll','2026-03-14 21:23:26','I\'m going to assume you\'re either frustrated, bored, or testing me again.\n\nIf you\'re genuinely stuck or don\'t know how to answer my question, just say that. If you\'re frustrated with this conversation, I get it — but I can actually help you, and I\'d rather do that than guess.\n\nSo: what work are you putting off?',NULL);
/*!40000 ALTER TABLE `MCIIInterventions` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `Predictions`
--

LOCK TABLES `Predictions` WRITE;
/*!40000 ALTER TABLE `Predictions` DISABLE KEYS */;
INSERT INTO `Predictions` VALUES (1,1,1,'2026-03-09','3window','medium',0.54,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 11, \"days_until_deadline\": 6}'),(4,3,3,'2026-03-09','3window','high',0.67,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 0, \"days_until_deadline\": 6}'),(30,1,1,'2026-03-10','3window','low',0.19,NULL,'{\"overdue_count\": 0, \"completion_rate\": 1.0, \"submitted_today\": 0, \"days_since_last_sub\": 12, \"days_until_deadline\": 5}'),(37,3,3,'2026-03-10','3window','high',0.65,'null','{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 1, \"days_until_deadline\": 5}'),(39,1,1,'2026-03-11','3window','low',0.19,NULL,'{\"overdue_count\": 0, \"completion_rate\": 1.0, \"submitted_today\": 0, \"days_since_last_sub\": 13, \"days_until_deadline\": 4}'),(51,1,1,'2026-03-12','3window','low',0.20,NULL,'{\"overdue_count\": 0, \"completion_rate\": 1.0, \"submitted_today\": 0, \"days_since_last_sub\": 14, \"days_until_deadline\": 3}'),(52,1,1,'2026-03-13','3window','low',0.20,'null','{\"overdue_count\": 0, \"completion_rate\": 1.0, \"submitted_today\": 0, \"days_since_last_sub\": 15, \"days_until_deadline\": 2}'),(53,3,3,'2026-03-13','3window','medium',0.64,'null','{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 4, \"days_until_deadline\": 2}'),(56,1,1,'2026-03-14','3window','low',0.20,NULL,'{\"overdue_count\": 0, \"completion_rate\": 1.0, \"submitted_today\": 0, \"days_since_last_sub\": 16, \"days_until_deadline\": 1}'),(57,4,4,'2026-03-14','3window','high',0.67,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 0, \"days_until_deadline\": 1}'),(58,5,5,'2026-03-14','3window','medium',0.48,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 0, \"days_until_deadline\": 1}'),(59,6,6,'2026-03-14','3window','high',0.67,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 0, \"days_until_deadline\": 1}'),(60,3,3,'2026-03-14','3window','medium',0.64,'null','{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 5, \"days_until_deadline\": 1}'),(61,7,7,'2026-03-14','3window','medium',0.48,NULL,'{\"overdue_count\": 0, \"completion_rate\": 0.0, \"submitted_today\": 0, \"days_since_last_sub\": 0, \"days_until_deadline\": 1}');
/*!40000 ALTER TABLE `Predictions` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `Students`
--

LOCK TABLES `Students` WRITE;
/*!40000 ALTER TABLE `Students` DISABLE KEYS */;
INSERT INTO `Students` VALUES (1,'jerry@alu.com','Jerry KoolAid','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-02-26','low','mixed',1,'2026-02-26 16:45:20','0798284094','1_27e3b802.jpg','Hi I\'m Jerry',NULL),(2,'j.agbaje@alustudent.com','Jeremiah Agbaje','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-02','low','early',0,'2026-03-02 01:00:07',NULL,NULL,NULL,NULL),(3,'j@alu.com','Jeremiah Agbaje','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-09','medium','lastminute',3,'2026-03-08 22:12:51',NULL,NULL,NULL,NULL),(4,'jason@alu.com','Jeremiah Jason','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-14','high','lastminute',0,'2026-03-13 21:33:30','08023144061','/media/profile_pics/4_97fe2448.png','sup',1),(5,'johnnyboy@gmail.com','John Doe ','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-14','medium','early',0,'2026-03-13 21:44:51',NULL,NULL,NULL,1),(6,'freeman@gmail.com','freeman','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-14','high','lastminute',0,'2026-03-13 21:47:18',NULL,NULL,NULL,NULL),(7,'jasonb@gmail.com','Jason bigham','5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5','2026-03-14','medium','early',0,'2026-03-14 18:44:25',NULL,NULL,NULL,NULL);
/*!40000 ALTER TABLE `Students` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `Surveys`
--

LOCK TABLES `Surveys` WRITE;
/*!40000 ALTER TABLE `Surveys` DISABLE KEYS */;
/*!40000 ALTER TABLE `Surveys` ENABLE KEYS */;
UNLOCK TABLES;

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
-- Dumping data for table `Tasks`
--

LOCK TABLES `Tasks` WRITE;
/*!40000 ALTER TABLE `Tasks` DISABLE KEYS */;
INSERT INTO `Tasks` VALUES (4,1,NULL,'Submit physics lab project ','','2026-03-06 22:07:00','2026-03-05 17:04:26','2026-03-09 20:30:35','completed',NULL,'personal',0),(5,1,1,'do demo','m,m,m,  m, , m','2026-03-14 22:11:00','2026-03-05 17:11:24','2026-03-09 17:22:51','completed',NULL,'personal',0),(6,1,NULL,' keep coding','','2026-03-19 01:29:00','2026-03-08 21:29:57','2026-03-09 17:22:47','completed',NULL,'personal',0),(7,1,NULL,'Test Task','','2026-03-16 12:00:00','2026-03-08 21:56:28','2026-03-09 17:22:49','completed',NULL,'personal',0),(8,1,1,'overdue','','2026-03-03 15:52:00','2026-03-09 11:52:29','2026-03-09 20:35:08','completed',NULL,'personal',0),(9,1,1,'overdue test 2 ','','2026-03-04 22:35:00','2026-03-09 18:35:35','2026-03-09 20:35:54','completed',NULL,'personal',0),(10,6,6,'submit physics report ',NULL,'2026-03-14 00:00:00','2026-03-13 21:49:01',NULL,'pending',NULL,'personal',0),(12,4,4,'Test Assignment',NULL,'2026-03-24 00:00:00','2026-03-13 22:11:06','2026-03-14 00:12:50','completed',1,'assignment',1),(13,5,5,'Test Assignment',NULL,'2026-03-24 00:00:00','2026-03-13 22:11:06',NULL,'pending',1,'assignment',1),(14,1,1,'do ethics assignment ',NULL,'2026-03-15 16:30:00','2026-03-14 12:18:13','2026-03-14 14:51:13','completed',NULL,'personal',0),(15,4,4,'Ethics assignment',NULL,'2026-03-10 23:59:00','2026-03-14 13:05:05','2026-03-14 15:05:52','completed',1,'assignment',1),(16,5,5,'Ethics assignment',NULL,'2026-03-10 23:59:00','2026-03-14 13:05:05',NULL,'pending',1,'assignment',1);
/*!40000 ALTER TABLE `Tasks` ENABLE KEYS */;
UNLOCK TABLES;

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

--
-- Dumping data for table `WeeklyBundles`
--

LOCK TABLES `WeeklyBundles` WRITE;
/*!40000 ALTER TABLE `WeeklyBundles` DISABLE KEYS */;
INSERT INTO `WeeklyBundles` VALUES (1,1,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-09 00:00:52'),(3,3,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-08 22:12:51'),(4,4,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-13 21:33:30'),(5,5,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-13 21:44:51'),(6,6,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-13 21:47:18'),(7,7,11,'2026-03-09','2026-03-15',0,0,0,0.000,0,0,NULL,'2026-03-14 18:44:25');
/*!40000 ALTER TABLE `WeeklyBundles` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-03-15 12:42:35

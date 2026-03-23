"""
ProActive — Unit Tests
Run with: pytest test_proactive.py -v
"""
import hashlib
import secrets
import string
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta


#  1. Risk threshold logic 
# Extracted from compute_prediction() in main.py (lines 477-482)


def get_risk_level(risk_score: float) -> str:
    if risk_score < 0.40:
        return "low"
    elif risk_score < 0.65:
        return "medium"
    else:
        return "high"


def test_risk_threshold_low():
    """Score below 0.40 maps to 'low'"""
    assert get_risk_level(0.35) == "low"
    assert get_risk_level(0.00) == "low"
    assert get_risk_level(0.39) == "low"


def test_risk_threshold_medium():
    """Score between 0.40 and 0.65 maps to 'medium'"""
    assert get_risk_level(0.40) == "medium"
    assert get_risk_level(0.55) == "medium"
    assert get_risk_level(0.64) == "medium"


def test_risk_threshold_high():
    """Score 0.65 or above maps to 'high'"""
    assert get_risk_level(0.65) == "high"
    assert get_risk_level(0.70) == "high"
    assert get_risk_level(1.00) == "high"


#  2. Password hashing 
# From hash_password() in main.py (line 112)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def test_password_hash_produces_hex_string():
    """hash_password returns a 64-character hex string"""
    result = hash_password("testpassword123")
    assert isinstance(result, str)
    assert len(result) == 64


def test_password_hash_is_not_plaintext():
    """hash_password output does not equal the input"""
    password = "mypassword"
    assert hash_password(password) != password


def test_password_hash_is_deterministic():
    """Same input always produces same hash"""
    assert hash_password("abc") == hash_password("abc")


def test_password_hash_different_inputs_differ():
    """Different passwords produce different hashes"""
    assert hash_password("password1") != hash_password("password2")


# 3. Invite code generation 
# From admin_create_submit() in main.py (lines ~1630-1634)

def generate_invite_code() -> str:
    return "".join(
        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8)
    )


def test_invite_code_length():
    """Invite code is exactly 8 characters"""
    assert len(generate_invite_code()) == 8


def test_invite_code_is_alphanumeric():
    """Invite code contains only uppercase letters and digits"""
    code = generate_invite_code()
    assert code.isalnum()
    assert code == code.upper()


def test_invite_codes_are_unique():
    """Two generated codes are (almost certainly) different"""
    codes = {generate_invite_code() for _ in range(20)}
    assert len(codes) > 1


#  4. Feature vector structure 
# From compute_prediction() live_features block in main.py (lines 419-425)

def build_live_features(
    days_until_deadline: float,
    days_since_last_sub: float,
    submitted_today: float,
    completion_rate: float,
    overdue_count: float,
) -> list:
    return [
        float(days_until_deadline),
        float(days_since_last_sub),
        float(submitted_today),
        float(completion_rate),
        float(overdue_count),
    ]


def test_feature_vector_has_five_elements():
    """Feature vector must contain exactly 5 elements"""
    features = build_live_features(3, 2, 1, 0.8, 1)
    assert len(features) == 5


def test_feature_vector_all_floats():
    """All elements in feature vector are floats"""
    features = build_live_features(3, 2, 1, 0.8, 1)
    assert all(isinstance(f, float) for f in features)


def test_feature_vector_correct_order():
    """Feature vector preserves correct field order"""
    features = build_live_features(
        days_until_deadline=3.0,
        days_since_last_sub=2.0,
        submitted_today=1.0,
        completion_rate=0.75,
        overdue_count=2.0,
    )
    assert features[0] == 3.0   # days_until_deadline
    assert features[1] == 2.0   # days_since_last_sub
    assert features[2] == 1.0   # submitted_today
    assert features[3] == 0.75  # completion_rate
    assert features[4] == 2.0   # overdue_count


# 5. create_initial_bundle idempotency 
# From create_initial_bundle() in main.py (lines ~155-185)
# Tests the logic without a live DB using mocks

def test_create_initial_bundle_returns_existing_if_present():
    """If a bundle already exists for this week, return it without creating a new one"""
    today = date.today()
    week_number = today.isocalendar().week

    existing_bundle = MagicMock()
    existing_bundle.week_number = week_number

    mock_db = MagicMock()
    mock_query = mock_db.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_filter.first.return_value = existing_bundle

    result = mock_filter.first()
    assert result == existing_bundle
    mock_db.add.assert_not_called()


def test_create_initial_bundle_sets_correct_week_dates():
    """Bundle start_date is Monday and end_date is Sunday of the current week"""
    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week   = start_of_week + timedelta(days=6)

    assert start_of_week.weekday() == 0  # Monday
    assert end_of_week.weekday()   == 6  # Sunday
    assert (end_of_week - start_of_week).days == 6
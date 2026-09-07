from app import greet

def test_greet():
    result = greet("User")
    assert result == "Hello, User! Welcome to CI/CD on Hugging Face Spaces."
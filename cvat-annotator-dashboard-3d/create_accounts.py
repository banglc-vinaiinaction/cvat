import requests
import json
import os

CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080")

def create_user_and_pat(username, password, email, role="user"):
    print(f"\n--- Creating account for {username} ({role}) ---")
    
    # 1. Register the user
    register_url = f"{CVAT_URL}/api/auth/register"
    user_data = {
        "username": username,
        "password1": password,
        "password2": password,
        "email": email,
        "first_name": username,
        "last_name": role
    }
    
    try:
        # Register user
        res = requests.post(register_url, json=user_data)
        if res.status_code == 201:
            print(f"✅ User {username} registered successfully.")
        elif res.status_code == 400 and 'username' in res.text:
            print(f"⚠️ User {username} already exists. Continuing to generate PAT.")
        else:
            print(f"❌ Failed to register user {username}: {res.text}")
            return None
            
        # 2. Login to get session/token to create PAT
        # CVAT login can be done via POST /api/auth/login
        login_url = f"{CVAT_URL}/api/auth/login"
        login_data = {"username": username, "password": password}
        session = requests.Session()
        login_res = session.post(login_url, json=login_data)
        
        if login_res.status_code != 200:
            print(f"❌ Failed to login as {username}: {login_res.text}")
            return None
            
        # Set CSRF Token if needed
        if 'csrftoken' in session.cookies:
            session.headers.update({'X-CSRFToken': session.cookies['csrftoken']})
            
        # 3. Create PAT
        pat_url = f"{CVAT_URL}/api/auth/access_tokens"
        pat_data = {"name": f"{username}-pat"}
        pat_res = session.post(pat_url, json=pat_data)
        
        if pat_res.status_code == 201:
            pat_key = pat_res.json().get('key')
            print(f"✅ PAT generated for {username}")
            return pat_key
        else:
            print(f"❌ Failed to generate PAT for {username}: {pat_res.text}")
            return None
            
    except Exception as e:
        print(f"Error connecting to CVAT: {e}")
        return None

if __name__ == "__main__":
    print("Setting up accounts...")
    
    accounts = [
        {"username": "reviewer1", "password": "Password123!", "email": "rev1@example.com", "role": "reviewer"},
        {"username": "annotator1", "password": "Password123!", "email": "ann1@example.com", "role": "annotator"},
        {"username": "annotator2", "password": "Password123!", "email": "ann2@example.com", "role": "annotator"},
    ]
    
    results = []
    
    for acc in accounts:
        pat = create_user_and_pat(acc["username"], acc["password"], acc["email"], acc["role"])
        if pat:
            results.append({
                "username": acc["username"],
                "password": acc["password"],
                "role": acc["role"],
                "PAT": pat
            })
            
    print("\n\n🎉 --- ACCOUNT SUMMARY --- 🎉")
    print(json.dumps(results, indent=4))
    print("Please save these credentials. The PATs are only shown once!")

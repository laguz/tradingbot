from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, public_key):
        self.id = public_key
        self.public_key = public_key

    @staticmethod
    def get(user_id):
        # In a real app, you'd fetch this from a DB.
        # For this P2P/Demo, we trust the ID if it's a valid key format,
        # or we could require it to be in a verified list.
        # For now, we allow any valid logged-in session to recreate the user object.
        if user_id:
            return User(user_id)
        return None

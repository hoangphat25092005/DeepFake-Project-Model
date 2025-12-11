import sys
sys.path.insert(0, '/mnt/mmlab2024nas/danh/phatlh/D3')

from database import DatabaseManager

if __name__ == "__main__":
    print("Initializing database...")
    db = DatabaseManager()
    db.create_tables()
    print("Database initialized!")
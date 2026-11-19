# config_test.py
from backend.config import get_settings

s = get_settings()
print("db_host:", s.db_host)
print("db_port:", s.db_port)
print("database_url:", s.database_url)
print("default_tenant_id:", s.default_tenant_id)
print("两次 get_settings 是同一对象(lru_cache):", get_settings() is get_settings())

#!/usr/bin/env python3
"""
Validate and fix .env file formatting issues.
"""

from pathlib import Path

def validate_env_file(env_path='.env'):
    """Check .env file for common issues."""
    
    print("=" * 80)
    print("ENV FILE VALIDATOR")
    print("=" * 80)
    
    path = Path(env_path)
    if not path.exists():
        print(f"❌ File not found: {env_path}")
        return
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    print(f"\nAnalyzing {env_path} ({len(lines)} lines)...\n")
    
    issues = []
    
    for i, line in enumerate(lines, 1):
        original = line
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
        
        # Check for equals sign
        if '=' not in line:
            issues.append((i, 'NO_EQUALS', f"Line {i}: Missing '=' sign"))
            continue
        
        # Check for spaces around equals
        if ' = ' in line or line.split('=')[0].rstrip() != line.split('=')[0]:
            issues.append((i, 'SPACES', f"Line {i}: Has spaces around '='"))
        
        # Check for unquoted special characters
        key, value = line.split('=', 1)
        value = value.rstrip('\n')
        
        if value and not (value.startswith('"') and value.endswith('"')):
            if any(char in value for char in ['#', ' ', '@', ':', '/', '?']):
                issues.append((i, 'UNQUOTED', f"Line {i}: Value with special chars should be quoted"))
    
    if not issues:
        print("✅ No issues found!")
        return True
    
    print(f"❌ Found {len(issues)} issue(s):\n")
    for i, issue_type, msg in issues:
        print(f"  {msg}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("""
1. Remove spaces around = sign:
   ❌ KEY = value
   ✅ KEY=value

2. Quote values with special characters:
   ❌ MONGODB_URI=mongodb://user:pass@host
   ✅ MONGODB_URI="mongodb://user:pass@host"

3. Move inline comments to separate lines:
   ❌ KEY=value # comment
   ✅ # comment
   ✅ KEY=value

4. Remove trailing whitespace
""")
    
    return False

if __name__ == '__main__':
    validate_env_file('.env')

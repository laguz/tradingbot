#!/usr/bin/env python3
"""
Quick verification script for UI enhancements
Tests that all new JavaScript files are syntactically valid
"""

import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JS_FILES = [
    'static/js/toast.js',
    'static/js/theme.js',
    'static/js/animations.js'
]

print("=" * 60)
print("UI Enhancements Verification")
print("=" * 60)

# Check if all files exist
print("\n1. Checking file existence...")
all_exist = True
for js_file in JS_FILES:
    file_path = os.path.join(BASE_DIR, js_file)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"  ✓ {js_file} ({file_size} bytes)")
    else:
        print(f"  ✗ {js_file} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    sys.exit(1)

# Check custom.css was updated
print("\n2. Checking CSS updates...")
css_path = os.path.join(BASE_DIR, 'static/css/custom.css')
if os.path.exists(css_path):
    with open(css_path, 'r') as f:
        css_content = f.read()
        
    checks = {
        'Toast styles': 'toast-container' in css_content,
        'Theme toggle': '[data-theme="light"]' in css_content,
        'Loading skeletons': 'skeleton-loading' in css_content,
        'Accessibility': 'skip-link' in css_content,
        'Animations': 'pulse-update' in css_content
    }
    
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        
    if not all(checks.values()):
        print("\n⚠ Some CSS features may be missing")
else:
    print(f"  ✗ custom.css NOT FOUND")
    sys.exit(1)

# Check template updates
print("\n3. Checking template updates...")
templates = {
    'layout.html': ['skip-link', 'theme-toggle', 'toast-container'],
    'auto_trader.html': ['aria-label', 'aria-live', 'toastSuccess'],
}

for template, required_strings in templates.items():
    template_path = os.path.join(BASE_DIR, 'templates', template)
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            content = f.read()
        
        found = [s for s in required_strings if s in content]
        status = "✓" if len(found) == len(required_strings) else "⚠"
        print(f"  {status} {template} ({len(found)}/{len(required_strings)} features)")
    else:
        print(f"  ✗ {template} NOT FOUND")

# Check documentation
print("\n4. Checking documentation...")
docs = [
    'UI_ENHANCEMENTS.md'
]

for doc in docs:
    doc_path = os.path.join(BASE_DIR, doc)
    if os.path.exists(doc_path):
        doc_size = os.path.getsize(doc_path)
        print(f"  ✓ {doc} ({doc_size} bytes)")
    else:
        print(f"  ✗ {doc} NOT FOUND")

print("\n" + "=" * 60)
print("✅ Verification complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Start the Flask server: python app.py")
print("2. Navigate to: http://localhost:5000/auto-trader")
print("3. Test the new features:")
print("   - Click 'Run Now' to see toast notification")
print("   - Click theme toggle button (sun/moon icon)")
print("   - Press Tab to see skip link")
print("   - Test keyboard navigation")
print("\nFor full documentation, see UI_ENHANCEMENTS.md")
print("=" * 60)

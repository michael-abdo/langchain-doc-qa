#!/usr/bin/env python3
"""
Import Dependency Validation Script
Checks for circular dependencies and validates all imports work correctly.
"""
import sys
import os
import importlib
from typing import Dict, Set, List
import traceback

def check_circular_dependencies() -> bool:
    """Check for circular import dependencies."""
    print("Checking for circular dependencies...")
    
    # Key modules to check
    modules_to_check = [
        'app.core.common',
        'app.core.exceptions', 
        'app.core.database',
        'app.core.config',
        'app.services.document_processor',
        'app.services.chunking',
        'app.services.vector_store',
        'app.services.metrics',
        'app.api.routes.documents',
        'app.api.routes.health',
        'app.api.routes.metrics'
    ]
    
    import_success = {}
    
    for module_name in modules_to_check:
        try:
            # Try to import each module
            module = importlib.import_module(module_name)
            import_success[module_name] = True
            print(f"SUCCESS {module_name}")
        except ImportError as e:
            import_success[module_name] = False
            print(f"FAILED {module_name}: {e}")
        except Exception as e:
            import_success[module_name] = False
            print(f"ERROR {module_name}: {e}")
    
    failed_modules = [name for name, success in import_success.items() if not success]
    
    if failed_modules:
        print(f"\nFAILED: {len(failed_modules)} modules failed to import:")
        for module in failed_modules:
            print(f"  - {module}")
        return False
    else:
        print(f"\nSUCCESS: All {len(modules_to_check)} modules imported successfully")
        return True

def validate_consolidation_utilities() -> bool:
    """Validate that consolidation utilities work correctly."""
    print("\nValidating consolidation utilities...")
    
    try:
        # Test BaseService
        from app.core.common import BaseService
        test_service = BaseService("test")
        assert test_service.service_name == "test"
        assert not test_service.is_initialized
        test_service.initialize()
        assert test_service.is_initialized
        print("SUCCESS BaseService works correctly")
        
        # Test CommonValidators
        from app.core.common import CommonValidators
        CommonValidators.validate_document_id("test123")
        CommonValidators.validate_content_not_empty(b"test content")
        CommonValidators.validate_text_extraction("This is a longer text for testing")
        print("SUCCESS CommonValidators work correctly")
        
        # Test ApiResponses
        from app.core.common import ApiResponses
        success_resp = ApiResponses.success({"test": "data"})
        assert success_resp["status"] == "success"
        error_resp = ApiResponses.error("test error")
        assert error_resp["status"] == "error"
        print("SUCCESS ApiResponses work correctly")
        
        # Test error handling utilities
        from app.core.exceptions import handle_service_error
        # Don't actually trigger the error, just verify it's importable
        print("SUCCESS Error handling utilities importable")
        
        # Test config accessor
        from app.core.config import config
        processing_config = config.processing_config
        assert isinstance(processing_config, dict)
        print("SUCCESS Config accessor works correctly")
        
        return True
        
    except Exception as e:
        print(f"FAILED Consolidation utility validation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("Starting import dependency validation...")
    
    # Change to project directory
    if not os.path.exists('app'):
        print("FAILED Must run from project root directory")
        sys.exit(1)
    
    # Add project to Python path
    sys.path.insert(0, os.getcwd())
    
    # Run checks
    circular_deps_ok = check_circular_dependencies()
    utilities_ok = validate_consolidation_utilities()
    
    print("\n" + "="*50)
    if circular_deps_ok and utilities_ok:
        print("SUCCESS All import dependency checks PASSED")
        print("READY to proceed with DRY refactoring")
        return True
    else:
        print("FAILED Import dependency checks FAILED")
        print("WARNING Fix issues before proceeding with refactoring")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Direct IE Service Test
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment variable to disable model loading
os.environ["SUBGRAPHRAG_DISABLE_MODEL_LOADING"] = "true"

async def test_ie_service():
    """Test IE service directly"""
    print("🧪 Testing IE service directly...")
    
    try:
        from src.app.services.information_extraction import get_information_extraction_service
        print("✅ Successfully imported IE service")
        
        # Get service instance
        ie_service = get_information_extraction_service()
        print("✅ Got IE service instance")
        
        # Test extract_triples (now async)
        print("🔄 Testing extract_triples...")
        result = await ie_service.extract_triples("Barack Obama was born in Hawaii.")
        print(f"✅ Extract triples completed")
        print(f"   Success: {result.success}")
        print(f"   Triples count: {len(result.triples)}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Error: {result.error_message}")
        
        if result.triples:
            print("   Triples:")
            for i, triple in enumerate(result.triples):
                print(f"     {i+1}. {triple}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ie_service())
    print(f"\n🏁 Test {'PASSED' if success else 'FAILED'}")
"""
Test script to verify OpinionMiner installation.
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    packages = [
        ('streamlit', 'Streamlit'),
        ('feedparser', 'Feedparser'),
        ('groq', 'Groq'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('bs4', 'BeautifulSoup4'),
        ('requests', 'Requests'),
        ('dotenv', 'Python-dotenv'),
    ]

    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")

    required_files = [
        'src/database/db_manager.py',
        'src/data_collection/reddit_rss.py',
        'src/data_collection/rss_scraper.py',
        'src/data_collection/hn_rss.py',
        'src/data_collection/orchestrator.py',
        'src/processing/quality_filter.py',
        'src/processing/topic_classifier.py',
        'src/processing/embedding_generator.py',
        'src/recommendation/bandit.py',
        'src/recommendation/feed_generator.py',
        'src/ui/app.py',
        'requirements.txt',
        '.env.example',
        'README.md',
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing.append(file)

    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def test_database():
    """Test database initialization."""
    print("\nTesting database...")

    try:
        from src.database.db_manager import DatabaseManager

        db = DatabaseManager()
        stats = db.get_database_stats()

        print(f"  ✓ Database initialized")
        print(f"    Total opinions: {stats['total_opinions']}")
        print(f"    Processed opinions: {stats['processed_opinions']}")
        print(f"    Total users: {stats['total_users']}")

        print("\n✅ Database working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Database error: {e}")
        return False


def test_environment():
    """Test environment configuration."""
    print("\nTesting environment...")

    if not os.path.exists('.env'):
        print("  ⚠️  .env file not found")
        print("    Copy .env.example to .env and add your Groq API key")
        print("    System will use fallback heuristics without API key")
        return True

    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv('GROQ_API_KEY')

    if groq_key and groq_key != 'your_groq_api_key_here':
        print("  ✓ GROQ_API_KEY configured")
        print("\n✅ Environment configured!")
        return True
    else:
        print("  ⚠️  GROQ_API_KEY not configured")
        print("    Add your Groq API key to .env")
        print("    Get free key at: https://console.groq.com/")
        return True


def main():
    """Run all tests."""
    print("="*60)
    print("OpinionMiner Installation Test")
    print("="*60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Database", test_database()))
    results.append(("Environment", test_environment()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to use OpinionMiner!")
        print("\nNext steps:")
        print("1. python -m src.data_collection.orchestrator")
        print("2. python process_opinions.py")
        print("3. streamlit run src/ui/app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

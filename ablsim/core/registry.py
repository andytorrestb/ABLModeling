from pathlib import Path
from typing import List, Optional
from ablsim.core.case import Case

class CaseRegistry:
    def __init__(self, cases_root: Path):
        self.cases_root = Path(cases_root).resolve()
        
    def find_cases(self) -> List[Case]:
        """
        Scan the cases directory for valid case folders (containing case.json).
        """
        cases = []
        if not self.cases_root.exists():
            return []
            
        for path in self.cases_root.iterdir():
            if path.is_dir() and (path / "case.json").exists():
                try:
                    cases.append(Case(path))
                except Exception as e:
                    # Log warning but skip invalid cases?
                    print(f"Warning: Skipping invalid case at {path}: {e}")
        return cases

    def get_case(self, name: str) -> Optional[Case]:
        """
        Retrieve a specific case by directory name.
        """
        case_path = self.cases_root / name
        if case_path.exists() and (case_path / "case.json").exists():
            return Case(case_path)
        return None

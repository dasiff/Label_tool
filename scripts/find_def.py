from pathlib import Path
b=Path('scripts/labeling_tool.py').read_bytes()
s=b.decode('utf-8','replace')
for i,l in enumerate(s.splitlines()):
    if 'def _build_segment_features' in l:
        print('FOUND LINE', i+1)
        start=max(0,i-20)
        end=min(len(s.splitlines()), i+20)
        print('\n'.join(s.splitlines()[start:end]))
        break
else:
    print('NOT FOUND')

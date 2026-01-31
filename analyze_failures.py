
import csv
import sys
import collections

# Fix for field size limit overflow on Windows
try:
    csv.field_size_limit(13107200)
except OverflowError:
    csv.field_size_limit(sys.maxsize)

stats = collections.defaultdict(lambda: collections.Counter())
item_15_examples = []

with open('failures_analysis.txt', 'w', encoding='utf-8') as outfile:
    try:
        with open('results/suspicious_boundaries_v9.csv', 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = row.get('item_id', 'Unknown')
                if not item:
                    item = 'Unknown'
                
                # Capture examples for Item 15
                if item == '15' and len(item_15_examples) < 5:
                    item_15_examples.append(row.get('item', 'N/A') + f" (Part: {row.get('item_part', 'N/A')})")
                
                flags_str = row.get('flags', '')
                if not flags_str:
                    continue
                
                current_flags = [f.strip() for f in flags_str.split(';') if f.strip()]
                for flag in current_flags:
                    stats[flag][item] += 1

        outfile.write(f"Item 15 Examples:\n")
        for ex in item_15_examples:
            outfile.write(f"  - {ex}\n")
        outfile.write("-" * 30 + "\n")
        
        # Print top 5 items for each failure mode found
        for flag in sorted(stats.keys()):
            item_counts = stats[flag]
            total_mode_count = sum(item_counts.values())
            outfile.write(f"\nFailure Mode: {flag}\n")
            outfile.write(f"  Total Occurrences: {total_mode_count}\n")
            outfile.write("  Top Affected Items:\n")
            for item, count in item_counts.most_common(5):
                pct = (count / total_mode_count) * 100
                outfile.write(f"    Item {item}: {count} ({pct:.1f}%)\n")

    except Exception as e:
        outfile.write(f"Error processing CSV: {e}\n")

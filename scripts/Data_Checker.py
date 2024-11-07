from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import calendar
import sys
import argparse

class StayCalculator:
    def __init__(self):
        self.trips: List[Tuple[datetime, datetime]] = []
        
    def add_trip(self, start_date: str, end_date: str) -> None:
        """Add a trip with dates in 'YYYY-MM-DD' format."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        self.trips.append((start, end))
        self.trips.sort(key=lambda x: x[0])  # Keep trips sorted by start date
        
    def load_trips_from_file(self, filename: str) -> List[str]:
        """
        Load trips from a text file with format 'YYYY-MM-DD -> YYYY-MM-DD' per line.
        Returns a list of any errors encountered.
        """
        errors = []
        line_number = 0
        
        try:
            with open(filename, 'r') as file:
                self.trips = []  # Clear existing trips
                for line in file:
                    line_number += 1
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    try:
                        start_str, end_str = line.split('->')
                        start_date = start_str.strip()
                        end_date = end_str.strip()
                        self.add_trip(start_date, end_date)
                    except ValueError as e:
                        errors.append(f"Line {line_number}: Invalid date format - {line}")
                    except Exception as e:
                        errors.append(f"Line {line_number}: Error processing line - {line}")
                        
        except FileNotFoundError:
            errors.append(f"File not found: {filename}")
        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")
            
        return errors
    
    def list_trips(self) -> List[str]:
        """Return a formatted list of all trips."""
        if not self.trips:
            return ["No trips recorded."]
            
        trip_list = []
        for i, (start, end) in enumerate(self.trips, 1):
            duration = (end - start).days + 1
            trip_list.append(
                f"Trip {i}: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} "
                f"({duration} days)"
            )
        return trip_list
        
    def days_outside_in_range(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate total days outside the country within a date range."""
        days = 0
        for trip_start, trip_end in self.trips:
            if trip_start <= end_date and trip_end >= start_date:
                overlap_start = max(trip_start, start_date)
                overlap_end = min(trip_end, end_date)
                days += (overlap_end - overlap_start).days + 1
        return days
    
    def check_rolling_window(self) -> Dict[str, List[str]]:
        """Check all possible 365-day windows for violations."""
        violations = []
        if not self.trips:
            return {"violations": []}
            
        # Create a day-by-day window from first trip to last trip
        start_date = self.trips[0][0]
        end_date = self.trips[-1][1]
        
        current_date = start_date
        while current_date <= end_date:
            window_end = current_date + timedelta(days=364)  # 365-day window
            days_outside = self.days_outside_in_range(current_date, window_end)
            
            if days_outside > 182:
                violation_msg = f"Violation: {days_outside} days outside during window "\
                              f"{current_date.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}"
                violations.append(violation_msg)
            
            current_date += timedelta(days=1)
            
        return {"violations": violations}
    
    def check_specific_year(self, year: int, tax_year: bool = False) -> Dict[str, int]:
        """
        Check days outside for a specific year.
        If tax_year is True, uses Apr 6 - Apr 5, otherwise uses calendar year.
        """
        if tax_year:
            start_date = datetime(year, 4, 6)
            end_date = datetime(year + 1, 4, 5)
        else:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
        days_outside = self.days_outside_in_range(start_date, end_date)
        return {
            "year": year,
            "tax_year": tax_year,
            "days_outside": days_outside,
            "within_limit": days_outside <= 182
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Visa Stay Calculator')
    parser.add_argument('-f', '--file', help='Input file containing trip dates')
    parser.add_argument('-y', '--year', type=int, help='Check specific calendar year')
    parser.add_argument('-t', '--tax-year', type=int, help='Check specific tax year')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Start interactive mode after processing file')
    return parser.parse_args()

def main():
    args = parse_args()
    calculator = StayCalculator()
    
    # Handle file input if provided
    if args.file:
        print(f"\nLoading trips from {args.file}")
        errors = calculator.load_trips_from_file(args.file)
        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(error)
            if not calculator.trips:  # If no trips were loaded successfully
                return
        else:
            print("Trips loaded successfully!")
        
        print("\nLoaded Trips:")
        for trip in calculator.list_trips():
            print(trip)
            
        # Check specific year if provided
        if args.year:
            result = calculator.check_specific_year(args.year)
            print(f"\nDays outside during {args.year}: {result['days_outside']}")
            print(f"Within 182-day limit: {result['within_limit']}")
            
        if args.tax_year:
            result = calculator.check_specific_year(args.tax_year, tax_year=True)
            print(f"\nDays outside during tax year {args.tax_year}/{args.tax_year+1}: "
                  f"{result['days_outside']}")
            print(f"Within 182-day limit: {result['within_limit']}")
            
        # Check for violations
        results = calculator.check_rolling_window()
        if results["violations"]:
            print("\nViolations found:")
            for violation in results["violations"]:
                print(violation)
        else:
            print("\nNo violations found!")
    
    # Enter interactive mode if requested or if no file was provided
    if args.interactive or not args.file:
        while True:
            print("\nVisa Stay Calculator")
            print("1. Load trips from file")
            print("2. Add a trip manually")
            print("3. List all trips")
            print("4. Check rolling window violations")
            print("5. Check specific year")
            print("6. Check tax year")
            print("7. Exit")
            
            choice = input("Enter your choice (1-7): ")
            
            if choice == "1":
                filename = input("Enter the filename: ")
                errors = calculator.load_trips_from_file(filename)
                if errors:
                    print("\nErrors encountered:")
                    for error in errors:
                        print(error)
                else:
                    print("Trips loaded successfully!")
                    
            elif choice == "2":
                start = input("Enter start date (YYYY-MM-DD): ")
                end = input("Enter end date (YYYY-MM-DD): ")
                try:
                    calculator.add_trip(start, end)
                    print("Trip added successfully!")
                except ValueError as e:
                    print(f"Error: {e}")
                    
            elif choice == "3":
                print("\nRecorded Trips:")
                for trip in calculator.list_trips():
                    print(trip)
                    
            elif choice == "4":
                results = calculator.check_rolling_window()
                if not results["violations"]:
                    print("No violations found!")
                else:
                    print("\nViolations found:")
                    for violation in results["violations"]:
                        print(violation)
                        
            elif choice == "5":
                year = int(input("Enter year to check: "))
                result = calculator.check_specific_year(year)
                print(f"\nDays outside during {year}: {result['days_outside']}")
                print(f"Within 182-day limit: {result['within_limit']}")
                
            elif choice == "6":
                year = int(input("Enter tax year to check (enter the starting year): "))
                result = calculator.check_specific_year(year, tax_year=True)
                print(f"\nDays outside during tax year {year}/{year+1}: {result['days_outside']}")
                print(f"Within 182-day limit: {result['within_limit']}")
                
            elif choice == "7":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
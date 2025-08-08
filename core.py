#!/usr/bin/env python3
"""
FinTerm (Finance Terminal)
Version: 1.0

A finance calculator that lives in your terminal!
Created by vsv ♡
"""

import argparse
import json
import math
import sys
import shlex
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import statsmodels.api as sm

# savestate/loadstate
import save_load

# user deletion
import delete_users

# ==================== DATA MODELS ====================

@dataclass
class Transaction:
    """Represents a financial transaction"""
    amount: float
    date: str
    category: str
    description: str
    type: str  # 'deposit' or 'withdrawal'


@dataclass
class Account:
    """Represents a bank account"""
    account_id: str
    account_type: str  # 'checking' or 'savings'
    balance: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)

    def debit(self, amount: float) -> bool:
        """Debit account if sufficient funds"""
        if self.balance >= amount:
            self.balance -= amount
            return True
        return False

    def credit(self, amount: float):
        """Credit account"""
        self.balance += amount


@dataclass
class UserProfile:
    """Represents a user profile"""
    user_id: str
    name: str
    accounts: List[Account] = field(default_factory=list)

    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID"""
        for account in self.accounts:
            if account.account_id == account_id:
                return account
        return None


# ==================== FINANCIAL CALCULATIONS ====================

class FinancialCalculations:
    """Core financial calculation functions"""

    @staticmethod
    def calculate_simple_interest(principal: float, rate: float, time: float) -> Dict[str, float]:
        """Calculate simple interest: I = P × r × t"""
        interest = principal * rate * time
        return {
            "interest": round(interest, 2),
            "final_amount": round(principal + interest, 2)
        }

    @staticmethod
    def calculate_compound_interest(principal: float, rate: float, time: float) -> Dict[str, float]:
        """Calculate compound interest: A = P × (1 + r)^t"""
        final_amount = principal * math.pow(1 + rate, time)
        interest = final_amount - principal
        return {
            "interest": round(interest, 2),
            "final_amount": round(final_amount, 2)
        }

    @staticmethod
    def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> Dict[str, float]:
        """Calculate monthly EMI for a loan"""
        monthly_rate = annual_rate / 12
        months = years * 12

        if monthly_rate == 0:
            emi = principal / months
        else:
            emi = principal * monthly_rate * math.pow(1 + monthly_rate, months) / (
                    math.pow(1 + monthly_rate, months) - 1)

        total_payment = emi * months
        total_interest = total_payment - principal

        return {
            "monthly_payment": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2)
        }

    @staticmethod
    def calculate_cagr(initial: float, final: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate"""
        if initial <= 0 or years <= 0:
            return 0.0
        cagr = (math.pow(final / initial, 1 / years) - 1) * 100
        return round(cagr, 2)

    @staticmethod
    def adjust_for_inflation(amount: float, rate: float, years: float) -> float:
        """Adjust amount for inflation"""
        real_value = amount * math.pow(100 / (100 + rate), years)
        return round(real_value, 2)

    @staticmethod
    def calculate_doubling_time(rate: float) -> float:
        """Calculate time to double money at given rate"""
        if rate <= 0:
            return float('inf')
        time = math.log(2) / math.log(1 + rate / 100)
        return round(time, 2)

    @staticmethod
    def weighted_average(weights: List[float], returns: List[float]) -> float:
        """Calculate weighted average return"""
        if len(weights) != len(returns):
            raise ValueError("Weights and returns must have same length")
        return round(sum(w * r for w, r in zip(weights, returns)), 4)

    @staticmethod
    def future_value_annuity(payment: float, rate: float, periods: int) -> float:
        """Calculate future value of annuity"""
        if rate == 0:
            return payment * periods
        fv = payment * ((math.pow(1 + rate, periods) - 1) / rate)
        return round(fv, 2)

    @staticmethod
    def present_value(cash_flow: float, rate: float, time: float) -> float:
        """Calculate present value"""
        pv = cash_flow / math.pow(1 + rate, time)
        return round(pv, 2)


# ==================== USER MANAGEMENT ====================

class UserManager:
    """Manages user profiles and accounts"""

    def __init__(self):
        self.users: Dict[str, UserProfile] = {}

    def create_user(self, user_id: str, name: str) -> UserProfile:
        """Create a new user profile"""
        user = UserProfile(user_id=user_id, name=name)
        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Fetch user by ID"""
        return self.users.get(user_id)

    def create_account(self, user_id: str, account_id: str, account_type: str = "checking") -> Optional[Account]:
        """Create account for new user"""
        user = self.get_user(user_id)
        if user:
            account = Account(account_id=account_id, account_type=account_type)
            user.accounts.append(account)
            return account
        return None


# ==================== TRANSACTION PROCESSING ====================

class TransactionProcessor:
    """Processes financial transactions"""

    @staticmethod
    def process_transaction(user_profile: UserProfile, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transaction and update account balance"""
        account_id = transaction.get('account_id',
                                     user_profile.accounts[0].account_id if user_profile.accounts else None)
        if not account_id:
            return {"status": "error", "message": "No account found"}

        account = user_profile.get_account(account_id)
        if not account:
            return {"status": "error", "message": "Account not found"}

        amount = transaction.get('amount', 0)
        trans_type = transaction.get('type', 'withdrawal' if amount < 0 else 'deposit')

        # Create transaction record
        trans_record = Transaction(
            amount=abs(amount),
            date=transaction.get('date', datetime.now().strftime('%Y-%m-%d')),
            category=transaction.get('category', 'general'),
            description=transaction.get('description', ''),
            type=trans_type
        )

        # Process transaction
        if trans_type == 'withdrawal' or amount < 0:
            if account.debit(abs(amount)):
                account.transactions.append(trans_record)
                return {
                    "status": "success",
                    "new_balance": account.balance,
                    "transaction_id": len(account.transactions)
                }
            else:
                return {"status": "error", "message": "Insufficient funds"}
        else:
            account.credit(abs(amount))
            account.transactions.append(trans_record)
            return {
                "status": "success",
                "new_balance": account.balance,
                "transaction_id": len(account.transactions)
            }


# ==================== ANALYTICS ====================

class Analytics:
    """Financial analytics and predictions"""

    @staticmethod
    def summarize_transactions(transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize transaction statistics"""
        if transactions_df.empty:
            return {"status": "no_data"}

        summary = {
            "total_transactions": len(transactions_df),
            "total_spent": float(transactions_df[transactions_df['amount'] < 0]['amount'].sum()),
            "total_income": float(transactions_df[transactions_df['amount'] > 0]['amount'].sum()),
            "average_transaction": float(transactions_df['amount'].mean()),
            "categories": transactions_df['category'].value_counts().to_dict()
        }

        # Monthly breakdown
        if 'date' in transactions_df.columns:
            transactions_df['month'] = pd.to_datetime(transactions_df['date']).dt.to_period('M')
            monthly = transactions_df.groupby('month')['amount'].sum()
            summary['monthly_totals'] = {str(k): float(v) for k, v in monthly.to_dict().items()}

        return summary

    @staticmethod
    def forecast_spending(transactions_df: pd.DataFrame, months_ahead: int = 3) -> List[float]:
        """Forecast future spending using linear regression"""
        if len(transactions_df) < 3:
            return []

        # Prepare data
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        monthly = transactions_df.groupby(transactions_df['date'].dt.to_period('M'))['amount'].sum()

        if len(monthly) < 2:
            return []

        # Create time series
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predict
        future_X = np.arange(len(monthly), len(monthly) + months_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)

        return [round(float(p), 2) for p in predictions]

    @staticmethod
    def recommend_portfolio(risk_profile: str, available_funds: float) -> Dict[str, float]:
        """Recommend portfolio allocation based on risk profile"""
        allocations = {
            "conservative": {"bonds": 0.70, "stocks": 0.20, "cash": 0.10},
            "moderate": {"bonds": 0.40, "stocks": 0.50, "cash": 0.10},
            "aggressive": {"bonds": 0.20, "stocks": 0.70, "alternatives": 0.10}
        }

        profile = allocations.get(risk_profile.lower(), allocations["moderate"])

        return {
            asset: round(allocation * available_funds, 2)
            for asset, allocation in profile.items()
        }

    @staticmethod
    def track_goal(current_savings: float, goal_amount: float, time_left_years: float) -> float:
        """Calculate required monthly savings to reach goal"""
        if time_left_years <= 0 or goal_amount <= current_savings:
            return 0.0

        months_left = time_left_years * 12
        remaining = goal_amount - current_savings
        monthly_required = remaining / months_left

        return round(monthly_required, 2)


# ==================== CLI INTERFACE & MORE ====================

def display_intro():
    """Print INTRO message"""
    intro = """
FinTerm (Finance Terminal)
Version: 1.0

A finance calculator that lives in your terminal!
Created by vsv ♡
"""
    print(intro)


def setup_parser():
    """Setup argument parser"""
    parser = argparse.ArgumentParser(
        description='FinTerm',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # User commands
    user_parser = subparsers.add_parser('user', help='User management')
    user_subparsers = user_parser.add_subparsers(dest='subcommand')

    create_user = user_subparsers.add_parser('create', help='Create user')
    create_user.add_argument('--user_id', required=True)
    create_user.add_argument('--name', required=True)

    # delete user
    delete_user = user_subparsers.add_parser('delete', help='Delete user')
    delete_user.add_argument('--user_id', required=True)

    list_users = user_subparsers.add_parser('list', help='List all users')

    # reset parser (after state_parser section)
    reset_parser = subparsers.add_parser('reset', help='System reset')
    reset_parser.add_argument('action', nargs='?', choices=['confirm', 'soft'], default='prompt')

    # Transaction commands
    trans_parser = subparsers.add_parser('transaction', help='Transaction processing')
    trans_subparsers = trans_parser.add_subparsers(dest='subcommand')

    process_trans = trans_subparsers.add_parser('process', help='Process transaction')
    process_trans.add_argument('--user_id', required=True)
    process_trans.add_argument('--amount', type=float, required=True)
    process_trans.add_argument('--category', default='general')
    process_trans.add_argument('--description', default='')

    # Calculate commands
    calc_parser = subparsers.add_parser('calculate', help='Financial calculations')
    calc_subparsers = calc_parser.add_subparsers(dest='subcommand')

    # Interest calculation
    interest = calc_subparsers.add_parser('interest', help='Calculate interest')
    interest.add_argument('--principal', type=float, required=True)
    interest.add_argument('--rate', type=float, required=True)
    interest.add_argument('--time', type=float, required=True)
    interest.add_argument('--type', choices=['simple', 'compound'], default='compound')

    # Loan calculation
    loan = calc_subparsers.add_parser('loan', help='Calculate loan payment')
    loan.add_argument('--principal', type=float, required=True)
    loan.add_argument('--rate', type=float, required=True)
    loan.add_argument('--years', type=int, required=True)

    # CAGR calculation
    cagr = calc_subparsers.add_parser('cagr', help='Calculate CAGR')
    cagr.add_argument('--initial', type=float, required=True)
    cagr.add_argument('--final', type=float, required=True)
    cagr.add_argument('--years', type=float, required=True)

    # Investment commands
    invest_parser = subparsers.add_parser('investment', help='Investment recommendations')
    invest_subparsers = invest_parser.add_subparsers(dest='subcommand')

    recommend = invest_subparsers.add_parser('recommend', help='Recommend portfolio')
    recommend.add_argument('--risk_profile', choices=['conservative', 'moderate', 'aggressive'], required=True)
    recommend.add_argument('--amount', type=float, required=True)

    # Behavior commands
    behavior_parser = subparsers.add_parser('behavior', help='Behavioral analysis')
    behavior_subparsers = behavior_parser.add_subparsers(dest='subcommand')

    analyze = behavior_subparsers.add_parser('analyze', help='Analyze transactions')
    analyze.add_argument('--user_id', required=True)

    # Savestate/Loadstate (State management commands)
    state_parser = subparsers.add_parser('state', help='Save/Load state')
    state_subparsers = state_parser.add_subparsers(dest='subcommand')

    save_state = state_subparsers.add_parser('save', help='Save current session')
    load_state = state_subparsers.add_parser('load', help='Load previous session')

    return parser


def process_command(command_str, parser, calc, user_manager, processor, analytics):
    """Process a single command and return result"""
    try:

        args = parser.parse_args(shlex.split(command_str))

        if args.command == 'user' and args.subcommand == 'create':
            user = user_manager.create_user(args.user_id, args.name)
            # Create default checking account
            user_manager.create_account(args.user_id, f"{args.user_id}_checking", "checking")
            result = {"user_id": user.user_id, "name": user.name, "accounts": len(user.accounts)}
            return json.dumps(result)

        elif args.subcommand == 'delete':
            result = delete_users.delete_user(user_manager, args.user_id)
            return json.dumps(result)

        elif args.subcommand == 'list':
            result = delete_users.list_users(user_manager)
            return json.dumps(result)

        elif args.command == 'calculate':
            if args.subcommand == 'interest':
                if args.type == 'simple':
                    result = calc.calculate_simple_interest(args.principal, args.rate, args.time)
                else:
                    result = calc.calculate_compound_interest(args.principal, args.rate, args.time)
                return json.dumps(result)

            elif args.subcommand == 'loan':
                result = calc.calculate_loan_payment(args.principal, args.rate, args.years)
                return json.dumps(result)

            elif args.subcommand == 'cagr':
                result = {"cagr": calc.calculate_cagr(args.initial, args.final, args.years)}
                return json.dumps(result)

        elif args.command == 'investment' and args.subcommand == 'recommend':
            result = analytics.recommend_portfolio(args.risk_profile, args.amount)
            return json.dumps(result)

        elif args.command == 'transaction' and args.subcommand == 'process':

            user = user_manager.get_user(args.user_id)
            if not user:
                user = user_manager.create_user(args.user_id, "Demo User")
                user_manager.create_account(args.user_id, f"{args.user_id}_checking", "checking")

            transaction = {
                'amount': args.amount,
                'category': args.category,
                'description': args.description,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            result = processor.process_transaction(user, transaction)
            return json.dumps(result)

        elif args.command == 'behavior' and args.subcommand == 'analyze':

            result = {
                "status": "success",
                "user_id": args.user_id,
                "analysis": {
                    "total_transactions": 0,
                    "average_daily_spending": 0.0,
                    "top_categories": []
                }
            }
            return json.dumps(result)

        elif args.command == 'state':
            if args.subcommand == 'save':
                if save_load.save_state(user_manager):
                    result = {"status": "success", "message": "Session saved successfully"}
                else:
                    result = {"status": "error", "message": "Failed to save session"}
                return json.dumps(result)

            elif args.subcommand == 'load':
                if save_load.load_state(user_manager):
                    result = {"status": "success", "message": "Session loaded successfully"}
                else:
                    result = {"status": "error", "message": "No saved session found"}
                return json.dumps(result)

        elif args.command == 'reset':
            if args.action == 'confirm':
                result = reset.reset_system(user_manager, confirm=True)
            elif args.action == 'soft':
                result = reset.soft_reset(user_manager)
            else:
                result = reset.reset_system(user_manager, confirm=False)
            return json.dumps(result)

        else:
            return json.dumps({"error": "Unknown command"})

    except SystemExit:
        # argparse tries to exit on error, so catch it lol
        return json.dumps({"error": "Invalid command. Type 'help' for available commands"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    """Main entry point with interactive loop"""
    # Display intro
    display_intro()

    # Initialize components
    parser = setup_parser()
    calc = FinancialCalculations()
    user_manager = UserManager()
    processor = TransactionProcessor()
    analytics = Analytics()

    # Interactive loop
    while True:
        try:
            # Get user input
            command = input("\nFinTerm> ").strip()

            # Check for exit
            if command.lower() in ['exit', 'quit', 'q']:
                print("Goodbye! ♡")
                break

            # Check for help
            elif command.lower() in ['help', 'h', '?']:
                display_intro()
                continue

            # Check for empty command
            elif not command:
                continue

            # Process command
            result = process_command(command, parser, calc, user_manager, processor, analytics)
            print(result)

        except KeyboardInterrupt:
            print("\nGoodbye! ♡")
            break
        except Exception as e:
            print(json.dumps({"error": f"Unexpected error: {str(e)}"}))


if __name__ == "__main__":
    main()
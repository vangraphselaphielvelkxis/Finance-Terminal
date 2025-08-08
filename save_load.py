#!/usr/bin/env python3
"""
Save/Load Module for FinTerm
"""

import pickle
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

# Default save file location (hidden file in user's home directory)
SAVE_FILE = Path.home() / '.finterm_savestate.pkl'


def save_state(user_manager) -> bool:
    """
    Save the current state of UserManager to disk

    Args:
        user_manager: The UserManager instance from core.py

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create a serializable version of the data
        save_data = {
            'version': '1.0.0',
            'users': {}
        }

        # Convert each user and their accounts to dict format
        for user_id, user in user_manager.users.items():
            user_data = {
                'user_id': user.user_id,
                'name': user.name,
                'accounts': []
            }

            # Convert each account
            for account in user.accounts:
                account_data = {
                    'account_id': account.account_id,
                    'account_type': account.account_type,
                    'balance': account.balance,
                    'transactions': []
                }

                # Convert each transaction
                for transaction in account.transactions:
                    trans_data = asdict(transaction)
                    account_data['transactions'].append(trans_data)

                user_data['accounts'].append(account_data)

            save_data['users'][user_id] = user_data

        # Save to file using pickle
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(save_data, f)

        return True

    except Exception as e:
        print(f"Error saving state: {e}")
        return False


def load_state(user_manager) -> bool:
    """
    Load a previously saved state into UserManager

    Args:
        user_manager: The UserManager instance from core.py

    Returns:
        bool: True if load successful, False otherwise
    """
    try:
        # Check if save file exists
        if not SAVE_FILE.exists():
            return False

        # Load data from file
        with open(SAVE_FILE, 'rb') as f:
            save_data = pickle.load(f)

        # Clear current users
        user_manager.users.clear()

        # Reconstruct users from saved data
        for user_id, user_data in save_data['users'].items():
            # Import necessary classes from core
            from core import UserProfile, Account, Transaction

            # Create user
            user = UserProfile(
                user_id=user_data['user_id'],
                name=user_data['name'],
                accounts=[]
            )

            # Recreate accounts
            for account_data in user_data['accounts']:
                account = Account(
                    account_id=account_data['account_id'],
                    account_type=account_data['account_type'],
                    balance=account_data['balance'],
                    transactions=[]
                )

                # Recreate transactions
                for trans_data in account_data['transactions']:
                    transaction = Transaction(
                        amount=trans_data['amount'],
                        date=trans_data['date'],
                        category=trans_data['category'],
                        description=trans_data['description'],
                        type=trans_data['type']
                    )
                    account.transactions.append(transaction)

                user.accounts.append(account)

            user_manager.users[user_id] = user

        return True

    except Exception as e:
        print(f"Error loading state: {e}")
        return False


def has_save() -> bool:
    """Check if a save file exists"""
    return SAVE_FILE.exists()


def delete_save() -> bool:
    """Delete the save file"""
    try:
        if SAVE_FILE.exists():
            os.remove(SAVE_FILE)
            return True
        return False
    except Exception as e:
        print(f"Error deleting save: {e}")
        return False


def get_save_info() -> Optional[Dict[str, Any]]:
    """Get information about the saved state without loading it"""
    try:
        if not SAVE_FILE.exists():
            return None

        # Get file stats
        stat = os.stat(SAVE_FILE)

        # Load just to count users (quick operation)
        with open(SAVE_FILE, 'rb') as f:
            save_data = pickle.load(f)

        return {
            'exists': True,
            'file_size': stat.st_size,
            'last_modified': stat.st_mtime,
            'user_count': len(save_data['users']),
            'version': save_data.get('version', 'unknown')
        }

    except Exception:
        return None


# Alternative JSON-based save/load (more readable but less efficient)
def save_state_json(user_manager) -> bool:
    """Save state as JSON (human-readable alternative)"""
    json_file = Path.home() / '.finterm_savestate.json'
    try:
        save_data = {
            'version': '1.0.0',
            'users': {}
        }

        for user_id, user in user_manager.users.items():
            user_data = {
                'user_id': user.user_id,
                'name': user.name,
                'accounts': []
            }

            for account in user.accounts:
                account_data = {
                    'account_id': account.account_id,
                    'account_type': account.account_type,
                    'balance': account.balance,
                    'transactions': [asdict(t) for t in account.transactions]
                }
                user_data['accounts'].append(account_data)

            save_data['users'][user_id] = user_data

        with open(json_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving JSON state: {e}")
        return False
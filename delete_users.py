#!/usr/bin/env python3
"""
User Deletion Module for FinTerm
"""

from typing import Dict, List, Optional, Any


def delete_user(user_manager, user_id: str) -> Dict[str, Any]:
    """
    Delete a specific user from the system

    Args:
        user_manager: The UserManager instance from core.py
        user_id: ID of the user to delete

    Returns:
        Dict with status and message
    """
    try:
        if user_id not in user_manager.users:
            return {
                "status": "error",
                "message": f"User '{user_id}' not found"
            }

        # Get user info before deletion for confirmation
        user = user_manager.users[user_id]
        user_name = user.name
        num_accounts = len(user.accounts)

        # Calculate total balance across all accounts
        total_balance = sum(account.balance for account in user.accounts)

        # Delete the user
        del user_manager.users[user_id]

        return {
            "status": "success",
            "message": f"User '{user_id}' ({user_name}) deleted",
            "deleted_info": {
                "user_id": user_id,
                "name": user_name,
                "accounts_deleted": num_accounts,
                "total_balance_lost": total_balance
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete user: {str(e)}"
        }


def list_users(user_manager) -> Dict[str, Any]:
    """
    List all users in the system

    Args:
        user_manager: The UserManager instance from core.py

    Returns:
        Dict with user list
    """
    try:
        if not user_manager.users:
            return {
                "status": "success",
                "users": [],
                "count": 0,
                "message": "No users in the system"
            }

        users_list = []
        for user_id, user in user_manager.users.items():
            user_info = {
                "user_id": user_id,
                "name": user.name,
                "accounts": len(user.accounts),
                "total_balance": sum(account.balance for account in user.accounts)
            }
            users_list.append(user_info)

        return {
            "status": "success",
            "users": users_list,
            "count": len(users_list)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list users: {str(e)}"
        }


def delete_all_users(user_manager) -> Dict[str, Any]:
    """
    Delete all users from the system (used by reset)

    Args:
        user_manager: The UserManager instance from core.py

    Returns:
        Dict with status and message
    """
    try:
        user_count = len(user_manager.users)

        if user_count == 0:
            return {
                "status": "success",
                "message": "No users to delete",
                "deleted_count": 0
            }

        # Clear all users
        user_manager.users.clear()

        return {
            "status": "success",
            "message": f"All {user_count} users deleted",
            "deleted_count": user_count
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete all users: {str(e)}"
        }


def get_user_info(user_manager, user_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific user

    Args:
        user_manager: The UserManager instance from core.py
        user_id: ID of the user

    Returns:
        Dict with user information
    """
    try:
        if user_id not in user_manager.users:
            return {
                "status": "error",
                "message": f"User '{user_id}' not found"
            }

        user = user_manager.users[user_id]

        # Gather account information
        accounts_info = []
        for account in user.accounts:
            account_info = {
                "account_id": account.account_id,
                "account_type": account.account_type,
                "balance": account.balance,
                "transaction_count": len(account.transactions)
            }
            accounts_info.append(account_info)

        return {
            "status": "success",
            "user_info": {
                "user_id": user.user_id,
                "name": user.name,
                "accounts": accounts_info,
                "total_accounts": len(user.accounts),
                "total_balance": sum(account.balance for account in user.accounts)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get user info: {str(e)}"
        }
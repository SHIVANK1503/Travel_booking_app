# Travel Booking System - Streamlit Web Application
# This project demonstrates all concepts from the Python Programming syllabus

import streamlit as st
import json
import csv
import sqlite3
import threading
import time
import logging
import re
import pickle
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import unittest
import io

# ==================== STREAMLIT PAGE CONFIG ====================
st.set_page_config(
    page_title="Travel Booking System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('travel_booking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== SESSION STATE INITIALIZATION ====================
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'booking_system' not in st.session_state:
    st.session_state.booking_system = None
if 'current_customer' not in st.session_state:
    st.session_state.current_customer = None

# ==================== ENUMS AND DATA CLASSES ====================
class TravelType(Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    PACKAGE = "package"

class BookingStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"

@dataclass
class Customer:
    """Customer data class demonstrating dataclass usage"""
    customer_id: str
    name: str
    email: str
    phone: str
    created_at: datetime.datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()

# ==================== CUSTOM EXCEPTIONS ====================
class TravelBookingException(Exception):
    """Base exception for travel booking system"""
    pass

class BookingNotFoundError(TravelBookingException):
    """Exception raised when booking is not found"""
    pass

class InvalidPaymentError(TravelBookingException):
    """Exception raised for invalid payment details"""
    pass

class InsufficientInventoryError(TravelBookingException):
    """Exception raised when inventory is insufficient"""
    pass

# ==================== ABSTRACT BASE CLASSES ====================
class BookingService(ABC):
    """Abstract base class for booking services"""
    
    @abstractmethod
    def create_booking(self, customer_id: str, details: Dict) -> str:
        pass
    
    @abstractmethod
    def cancel_booking(self, booking_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_booking_details(self, booking_id: str) -> Dict:
        pass

# ==================== DATABASE MANAGER ====================
class DatabaseManager:
    """Handles all database operations using SQLite"""
    
    def __init__(self, db_name: str = "travel_booking.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                
                # Customers table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS customers (
                        customer_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        phone TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Bookings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bookings (
                        booking_id TEXT PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        travel_type TEXT NOT NULL,
                        details TEXT NOT NULL,
                        amount REAL NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                ''')
                
                # Inventory table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS inventory (
                        item_id TEXT PRIMARY KEY,
                        item_type TEXT NOT NULL,
                        name TEXT NOT NULL,
                        available_quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        details TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def execute_query(self, query: str, params: tuple = None):
        """Execute a database query"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            raise

# ==================== DECORATORS ====================
def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def validate_customer_data(func):
    """Decorator to validate customer data"""
    def wrapper(*args, **kwargs):
        if len(args) > 1 and isinstance(args[1], dict):
            customer_data = args[1]
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if 'email' in customer_data and not re.match(email_pattern, customer_data['email']):
                raise ValueError("Invalid email format")
            if 'phone' in customer_data and len(customer_data['phone']) < 10:
                raise ValueError("Invalid phone number")
        return func(*args, **kwargs)
    return wrapper

# ==================== UTILITY FUNCTIONS ====================
def generate_id(prefix: str = "TB") -> str:
    """Generate unique ID using lambda and random"""
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"{prefix}{timestamp}{random_num}"

# Lambda functions for data processing
calculate_discount = lambda amount, percentage: amount * (percentage / 100)
format_currency = lambda amount: f"₹{amount:,.2f}"
is_weekend = lambda date: date.weekday() >= 5

# ==================== FLIGHT BOOKING SERVICE ====================
class FlightBookingService(BookingService):
    """Flight booking service implementation"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.lock = threading.Lock()
    
    @log_execution_time
    def create_booking(self, customer_id: str, details: Dict) -> str:
        """Create a new flight booking"""
        with self.lock:
            booking_id = generate_id("FL")
            
            # Validate flight details
            required_fields = ['from_city', 'to_city', 'departure_date', 'passengers']
            if not all(field in details for field in required_fields):
                raise ValueError("Missing required flight booking fields")
            
            # Calculate amount
            base_price = 5000
            passengers = details['passengers']
            total_amount = base_price * passengers
            
            # Apply weekend surcharge
            departure_date = datetime.datetime.strptime(details['departure_date'], '%Y-%m-%d')
            if is_weekend(departure_date):
                total_amount += calculate_discount(total_amount, 10)
            
            # Store booking
            booking_details = json.dumps(details)
            query = '''
                INSERT INTO bookings (booking_id, customer_id, travel_type, details, amount, status)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            self.db_manager.execute_query(
                query, 
                (booking_id, customer_id, TravelType.FLIGHT.value, booking_details, total_amount, BookingStatus.PENDING.value)
            )
            
            logger.info(f"Flight booking created: {booking_id}")
            return booking_id
    
    def cancel_booking(self, booking_id: str) -> bool:
        """Cancel a flight booking"""
        try:
            query = "UPDATE bookings SET status = ? WHERE booking_id = ? AND travel_type = ?"
            self.db_manager.execute_query(
                query, 
                (BookingStatus.CANCELLED.value, booking_id, TravelType.FLIGHT.value)
            )
            logger.info(f"Flight booking cancelled: {booking_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling booking {booking_id}: {e}")
            return False
    
    def get_booking_details(self, booking_id: str) -> Dict:
        """Get flight booking details"""
        query = "SELECT * FROM bookings WHERE booking_id = ? AND travel_type = ?"
        result = self.db_manager.execute_query(query, (booking_id, TravelType.FLIGHT.value))
        
        if not result:
            raise BookingNotFoundError(f"Flight booking {booking_id} not found")
        
        booking = result[0]
        return {
            'booking_id': booking[0],
            'customer_id': booking[1],
            'travel_type': booking[2],
            'details': json.loads(booking[3]),
            'amount': booking[4],
            'status': booking[5],
            'created_at': booking[6]
        }

# ==================== HOTEL BOOKING SERVICE ====================
class HotelBookingService(BookingService):
    """Hotel booking service implementation"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.lock = threading.Lock()
    
    @log_execution_time
    def create_booking(self, customer_id: str, details: Dict) -> str:
        """Create a new hotel booking"""
        with self.lock:
            booking_id = generate_id("HT")
            
            # Validate hotel details
            required_fields = ['hotel_name', 'city', 'check_in', 'check_out', 'rooms']
            if not all(field in details for field in required_fields):
                raise ValueError("Missing required hotel booking fields")
            
            # Calculate stay duration and amount
            check_in = datetime.datetime.strptime(details['check_in'], '%Y-%m-%d')
            check_out = datetime.datetime.strptime(details['check_out'], '%Y-%m-%d')
            nights = (check_out - check_in).days
            
            if nights <= 0:
                raise ValueError("Check-out date must be after check-in date")
            
            room_rate = 3000
            total_amount = room_rate * details['rooms'] * nights
            
            # Store booking
            booking_details = json.dumps(details)
            query = '''
                INSERT INTO bookings (booking_id, customer_id, travel_type, details, amount, status)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            self.db_manager.execute_query(
                query,
                (booking_id, customer_id, TravelType.HOTEL.value, booking_details, total_amount, BookingStatus.PENDING.value)
            )
            
            logger.info(f"Hotel booking created: {booking_id}")
            return booking_id
    
    def cancel_booking(self, booking_id: str) -> bool:
        """Cancel a hotel booking"""
        try:
            query = "UPDATE bookings SET status = ? WHERE booking_id = ? AND travel_type = ?"
            self.db_manager.execute_query(
                query,
                (BookingStatus.CANCELLED.value, booking_id, TravelType.HOTEL.value)
            )
            logger.info(f"Hotel booking cancelled: {booking_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling booking {booking_id}: {e}")
            return False
    
    def get_booking_details(self, booking_id: str) -> Dict:
        """Get hotel booking details"""
        query = "SELECT * FROM bookings WHERE booking_id = ? AND travel_type = ?"
        result = self.db_manager.execute_query(query, (booking_id, TravelType.HOTEL.value))
        
        if not result:
            raise BookingNotFoundError(f"Hotel booking {booking_id} not found")
        
        booking = result[0]
        return {
            'booking_id': booking[0],
            'customer_id': booking[1],
            'travel_type': booking[2],
            'details': json.loads(booking[3]),
            'amount': booking[4],
            'status': booking[5],
            'created_at': booking[6]
        }

# ==================== CUSTOMER MANAGER ====================
class CustomerManager:
    """Manages customer operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    @validate_customer_data
    def register_customer(self, customer_data: Dict) -> str:
        """Register a new customer"""
        try:
            customer_id = generate_id("CUST")
            query = '''
                INSERT INTO customers (customer_id, name, email, phone)
                VALUES (?, ?, ?, ?)
            '''
            self.db_manager.execute_query(
                query,
                (customer_id, customer_data['name'], customer_data['email'], customer_data['phone'])
            )
            logger.info(f"Customer registered: {customer_id}")
            return customer_id
        except sqlite3.IntegrityError:
            raise ValueError("Customer with this email already exists")
    
    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """Get customer by ID"""
        query = "SELECT * FROM customers WHERE customer_id = ?"
        result = self.db_manager.execute_query(query, (customer_id,))
        
        if result:
            customer_data = result[0]
            return Customer(
                customer_id=customer_data[0],
                name=customer_data[1],
                email=customer_data[2],
                phone=customer_data[3],
                created_at=datetime.datetime.fromisoformat(customer_data[4])
            )
        return None
    
    def get_all_customers(self) -> List[Customer]:
        """Get all customers"""
        query = "SELECT * FROM customers"
        results = self.db_manager.execute_query(query)
        
        customers = []
        for customer_data in results:
            customers.append(Customer(
                customer_id=customer_data[0],
                name=customer_data[1],
                email=customer_data[2],
                phone=customer_data[3],
                created_at=datetime.datetime.fromisoformat(customer_data[4])
            ))
        return customers

# ==================== ANALYTICS ENGINE ====================
class AnalyticsEngine:
    """Analytics and reporting using data processing concepts"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_booking_statistics(self) -> Dict:
        """Get comprehensive booking statistics"""
        query = "SELECT travel_type, amount, status, created_at FROM bookings"
        bookings = self.db_manager.execute_query(query)
        
        if not bookings:
            return {"message": "No bookings found"}
        
        # Convert to list of dictionaries
        booking_data = [
            {
                'travel_type': booking[0],
                'amount': booking[1],
                'status': booking[2],
                'created_at': booking[3]
            }
            for booking in bookings
        ]
        
        # Using filter, map, and reduce concepts
        total_bookings = len(booking_data)
        confirmed_bookings = list(filter(lambda x: x['status'] == 'confirmed', booking_data))
        cancelled_bookings = list(filter(lambda x: x['status'] == 'cancelled', booking_data))
        
        # Calculate revenue
        total_revenue = sum(map(lambda x: x['amount'], confirmed_bookings))
        average_booking_value = total_revenue / len(confirmed_bookings) if confirmed_bookings else 0
        
        # Group by travel type
        travel_type_stats = {}
        for booking in booking_data:
            travel_type = booking['travel_type']
            if travel_type not in travel_type_stats:
                travel_type_stats[travel_type] = {'count': 0, 'revenue': 0}
            
            travel_type_stats[travel_type]['count'] += 1
            if booking['status'] == 'confirmed':
                travel_type_stats[travel_type]['revenue'] += booking['amount']
        
        return {
            'total_bookings': total_bookings,
            'confirmed_bookings': len(confirmed_bookings),
            'cancelled_bookings': len(cancelled_bookings),
            'total_revenue': total_revenue,
            'average_booking_value': average_booking_value,
            'travel_type_breakdown': travel_type_stats,
            'booking_data': booking_data
        }
    
    def get_pandas_dataframe(self) -> pd.DataFrame:
        """Get bookings as pandas DataFrame for analysis"""
        query = """
            SELECT b.booking_id, b.customer_id, c.name, b.travel_type, 
                   b.amount, b.status, b.created_at
            FROM bookings b
            JOIN customers c ON b.customer_id = c.customer_id
        """
        results = self.db_manager.execute_query(query)
        
        if results:
            df = pd.DataFrame(results, columns=[
                'Booking ID', 'Customer ID', 'Customer Name', 'Travel Type',
                'Amount', 'Status', 'Created At'
            ])
            df['Amount'] = pd.to_numeric(df['Amount'])
            df['Created At'] = pd.to_datetime(df['Created At'])
            return df
        else:
            return pd.DataFrame()

# ==================== PAYMENT PROCESSOR ====================
class PaymentProcessor:
    """Handles payment processing"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def process_payment(self, booking_id: str, payment_details: Dict) -> bool:
        """Process payment for a booking"""
        try:
            # Validate payment details using regex
            card_pattern = r'^\d{16}$'
            cvv_pattern = r'^\d{3}$'
            
            if not re.match(card_pattern, payment_details.get('card_number', '')):
                raise InvalidPaymentError("Invalid card number format")
            
            if not re.match(cvv_pattern, payment_details.get('cvv', '')):
                raise InvalidPaymentError("Invalid CVV format")
            
            # Simulate payment processing
            time.sleep(1)
            
            # 90% success rate simulation
            if random.random() > 0.1:
                # Update booking status to confirmed
                query = "UPDATE bookings SET status = ? WHERE booking_id = ?"
                self.db_manager.execute_query(query, (BookingStatus.CONFIRMED.value, booking_id))
                logger.info(f"Payment successful for booking {booking_id}")
                return True
            else:
                raise InvalidPaymentError("Payment gateway error")
                
        except Exception as e:
            logger.error(f"Payment failed for booking {booking_id}: {e}")
            raise

# ==================== MAIN TRAVEL BOOKING SYSTEM ====================
class TravelBookingSystem:
    """Main travel booking system orchestrating all components"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.customer_manager = CustomerManager(self.db_manager)
        self.flight_service = FlightBookingService(self.db_manager)
        self.hotel_service = HotelBookingService(self.db_manager)
        self.payment_processor = PaymentProcessor(self.db_manager)
        self.analytics = AnalyticsEngine(self.db_manager)
        
        # Initialize sample data
        self._initialize_sample_data()
        logger.info("Travel Booking System initialized")
    
    def _initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        try:
            # Check if sample data already exists
            query = "SELECT COUNT(*) FROM customers"
            result = self.db_manager.execute_query(query)
            
            if result[0][0] == 0:  # No customers exist
                # Add sample customers
                sample_customers = [
                    {'name': 'John Doe', 'email': 'john@example.com', 'phone': '9876543210'},
                    {'name': 'Jane Smith', 'email': 'jane@example.com', 'phone': '9876543211'},
                    {'name': 'Mike Johnson', 'email': 'mike@example.com', 'phone': '9876543212'}
                ]
                
                for customer_data in sample_customers:
                    try:
                        self.customer_manager.register_customer(customer_data)
                    except Exception as e:
                        logger.warning(f"Could not create sample customer: {e}")
        
        except Exception as e:
            logger.warning(f"Could not initialize sample data: {e}")
    
    def register_customer(self, name: str, email: str, phone: str) -> str:
        """Register a new customer"""
        customer_data = {
            'name': name,
            'email': email,
            'phone': phone
        }
        return self.customer_manager.register_customer(customer_data)
    
    def book_flight(self, customer_id: str, from_city: str, to_city: str, 
                   departure_date: str, passengers: int = 1) -> Dict:
        """Book a flight"""
        try:
            customer = self.customer_manager.get_customer(customer_id)
            if not customer:
                raise ValueError("Customer not found")
            
            flight_details = {
                'from_city': from_city,
                'to_city': to_city,
                'departure_date': departure_date,
                'passengers': passengers
            }
            
            booking_id = self.flight_service.create_booking(customer_id, flight_details)
            
            return {
                'booking_id': booking_id,
                'status': 'success',
                'message': 'Flight booked successfully'
            }
            
        except Exception as e:
            logger.error(f"Flight booking error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def book_hotel(self, customer_id: str, hotel_name: str, city: str,
                  check_in: str, check_out: str, rooms: int = 1) -> Dict:
        """Book a hotel"""
        try:
            customer = self.customer_manager.get_customer(customer_id)
            if not customer:
                raise ValueError("Customer not found")
            
            hotel_details = {
                'hotel_name': hotel_name,
                'city': city,
                'check_in': check_in,
                'check_out': check_out,
                'rooms': rooms
            }
            
            booking_id = self.hotel_service.create_booking(customer_id, hotel_details)
            
            return {
                'booking_id': booking_id,
                'status': 'success',
                'message': 'Hotel booked successfully'
            }
            
        except Exception as e:
            logger.error(f"Hotel booking error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# ==================== STREAMLIT APPLICATION ====================
def initialize_system():
    """Initialize the booking system"""
    if st.session_state.booking_system is None:
        st.session_state.booking_system = TravelBookingSystem()
        st.session_state.db_manager = st.session_state.booking_system.db_manager

def main():
    """Main Streamlit application"""
    
    # Initialize system
    initialize_system()
    
    # Header
    st.title("🌍 Travel Booking System")
    st.markdown("### Your Online Campanion")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "👤 Customer Management", "✈️ Flight Booking", 
         "🏨 Hotel Booking", "💳 Payment Processing", "📊 Analytics", 
         "🔧 System Management"]
    )
    
    # Dashboard Page
    if page == "🏠 Dashboard":
        dashboard_page()
    
    # Customer Management Page
    elif page == "👤 Customer Management":
        customer_management_page()
    
    # Flight Booking Page
    elif page == "✈️ Flight Booking":
        flight_booking_page()
    
    # Hotel Booking Page
    elif page == "🏨 Hotel Booking":
        hotel_booking_page()
    
    # Payment Processing Page
    elif page == "💳 Payment Processing":
        payment_processing_page()
    
    # Analytics Page
    elif page == "📊 Analytics":
        analytics_page()
    
    # System Management Page
    elif page == "🔧 System Management":
        system_management_page()

def dashboard_page():
    """Dashboard page showing system overview"""
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get statistics
    stats = st.session_state.booking_system.analytics.get_booking_statistics()
    
    if "message" not in stats:
        with col1:
            st.metric("Total Bookings", stats['total_bookings'])
        
        with col2:
            st.metric("Confirmed Bookings", stats['confirmed_bookings'])
        
        with col3:
            st.metric("Cancelled Bookings", stats['cancelled_bookings'])
        
        with col4:
            st.metric("Total Revenue", format_currency(stats['total_revenue']))
        
        # Recent bookings
        st.subheader("Recent Bookings")
        df = st.session_state.booking_system.analytics.get_pandas_dataframe()
        if not df.empty:
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No bookings found")
    else:
        st.info("No booking data available yet")
    
    # System status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Database: Online")
        st.success("✅ Booking Services: Active")
    
    with col2:
        st.success("✅ Payment Processor: Active")
        st.success("✅ Analytics Engine: Running")

def customer_management_page():
    """Customer management page"""
    st.header("Customer Management")
    
    tab1, tab2, tab3 = st.tabs(["Register Customer", "View Customers", "Customer Search"])
    
    with tab1:
        st.subheader("Register New Customer")
        
        with st.form("customer_registration"):
            name = st.text_input("Full Name", placeholder="Enter customer's full name")
            email = st.text_input("Email", placeholder="customer@example.com")
            phone = st.text_input("Phone Number", placeholder="10-digit phone number")
            
            if st.form_submit_button("Register Customer"):
                if name and email and phone:
                    try:
                        customer_id = st.session_state.booking_system.register_customer(name, email, phone)
                        st.success(f"Customer registered successfully! Customer ID: {customer_id}")
                        st.session_state.current_customer = customer_id
                    except Exception as e:
                        st.error(f"Registration failed: {e}")
                else:
                    st.error("Please fill all fields")
    
    with tab2:
        st.subheader("All Customers")
        customers = st.session_state.booking_system.customer_manager.get_all_customers()
        
        if customers:
            # Convert to DataFrame for display
            customer_data = []
            for customer in customers:
                customer_data.append({
                    'Customer ID': customer.customer_id,
                    'Name': customer.name,
                    'Email': customer.email,
                    'Phone': customer.phone,
                    'Registration Date': customer.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            df = pd.DataFrame(customer_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No customers registered yet")
    
    with tab3:
        st.subheader("Search Customer")
        search_id = st.text_input("Enter Customer ID")
        
        if st.button("Search") and search_id:
            try:
                customer = st.session_state.booking_system.customer_manager.get_customer(search_id)
                if customer:
                    st.success(f"Customer Found: {customer.name}")
                    st.json({
                        'Customer ID': customer.customer_id,
                        'Name': customer.name,
                        'Email': customer.email,
                        'Phone': customer.phone,
                        'Registration Date': customer.created_at.isoformat()
                    })
                    st.session_state.current_customer = search_id
                else:
                    st.error("Customer not found")
            except Exception as e:
                st.error(f"Search failed: {e}")

def flight_booking_page():
    """Flight booking page"""
    st.header("Flight Booking")
    
    # Customer selection
    customers = st.session_state.booking_system.customer_manager.get_all_customers()
    
    if not customers:
        st.warning("No customers registered. Please register a customer first.")
        return
    
    customer_options = {f"{c.name} ({c.customer_id})": c.customer_id for c in customers}
    selected_customer = st.selectbox("Select Customer", options=list(customer_options.keys()))
    customer_id = customer_options[selected_customer]
    
    with st.form("flight_booking"):
        col1, col2 = st.columns(2)
        
        with col1:
            from_city = st.selectbox("From City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
            departure_date = st.date_input("Departure Date", min_value=datetime.date.today())
        
        with col2:
            to_city = st.selectbox("To City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
            passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
        
        if st.form_submit_button("Book Flight"):
            if from_city == to_city:
                st.error("From City and To City cannot be the same")
            else:
                result = st.session_state.booking_system.book_flight(
                    customer_id, from_city, to_city, str(departure_date), passengers
                )
                if result['status'] == 'success':
                    st.success(f"Booking successful! ID: {result['booking_id']}")
                else:
                    st.error(result['message'])


# ==================== HOTEL BOOKING PAGE ====================
def hotel_booking_page():
    """Hotel booking page"""
    st.header("Hotel Booking")
    
    customers = st.session_state.booking_system.customer_manager.get_all_customers()
    if not customers:
        st.warning("No customers registered. Please register a customer first.")
        return
    
    customer_options = {f"{c.name} ({c.customer_id})": c.customer_id for c in customers}
    selected_customer = st.selectbox("Select Customer", options=list(customer_options.keys()))
    customer_id = customer_options[selected_customer]
    
    with st.form("hotel_booking"):
        col1, col2 = st.columns(2)
        
        with col1:
            hotel_name = st.text_input("Hotel Name")
            city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
            check_in = st.date_input("Check-in Date", min_value=datetime.date.today())
        
        with col2:
            check_out = st.date_input("Check-out Date", min_value=datetime.date.today() + datetime.timedelta(days=1))
            rooms = st.number_input("Number of Rooms", min_value=1, max_value=5, value=1)
        
        if st.form_submit_button("Book Hotel"):
            if not hotel_name.strip():
                st.error("Please enter a hotel name.")
            elif check_out <= check_in:
                st.error("Check-out date must be after Check-in date")
            else:
                result = st.session_state.booking_system.book_hotel(
                    customer_id, hotel_name.strip(), city, str(check_in), str(check_out), rooms
                )
                if result['status'] == 'success':
                    st.success(f"Booking successful! ID: {result['booking_id']}")
                else:
                    st.error(result['message'])


# ==================== PAYMENT PROCESSING PAGE ====================
def payment_processing_page():
    """Payment processing page"""
    st.header("Payment Processing")
    
    booking_id = st.text_input("Enter Booking ID")
    
    with st.form("payment_form"):
        card_number = st.text_input("Card Number (16 digits)")
        cvv = st.text_input("CVV (3 digits)", type="password")
        expiry = st.text_input("Expiry Date (MM/YY)")
        
        if st.form_submit_button("Pay Now"):
            if not booking_id.strip():
                st.error("Please enter a booking ID.")
            else:
                try:
                    success = st.session_state.booking_system.payment_processor.process_payment(
                        booking_id.strip(),
                        {"card_number": card_number.strip(), "cvv": cvv.strip(), "expiry": expiry.strip()}
                    )
                    if success:
                        st.success("✅ Payment Successful! Booking confirmed.")
                    else:
                        st.error("❌ Payment Failed")
                except Exception as e:
                    st.error(f"Payment failed: {e}")


# ==================== ANALYTICS PAGE ====================
def analytics_page():
    """Analytics & Reports"""
    st.header("Analytics & Reports")
    
    stats = st.session_state.booking_system.analytics.get_booking_statistics()
    if "message" in stats:
        st.info("No booking data available")
        return
    
    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Bookings", stats['total_bookings'])
    with c2:
        st.metric("Confirmed", stats['confirmed_bookings'])
    with c3:
        st.metric("Cancelled", stats['cancelled_bookings'])
    with c4:
        st.metric("Revenue", format_currency(stats['total_revenue']))
    
    # Raw table
    df = st.session_state.booking_system.analytics.get_pandas_dataframe()
    if not df.empty:
        st.subheader("Bookings Data")
        st.dataframe(df.sort_values("Created At", ascending=False), use_container_width=True)
        
        # Charts
        st.subheader("Visualizations")
        fig1 = px.bar(df, x="Travel Type", y="Amount", color="Status", barmode="group", title="Amount by Travel Type & Status")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.pie(df, names="Travel Type", values="Amount", title="Revenue Share by Travel Type")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No bookings found for visualization")


# ==================== SYSTEM MANAGEMENT PAGE ====================
def system_management_page():
    """System management page"""
    st.header("System Management")
    
    if st.button("Reset Database"):
        # Recreate a fresh system (keeps your code intact, just resets data)
        st.session_state.booking_system = TravelBookingSystem()
        st.session_state.db_manager = st.session_state.booking_system.db_manager
        st.success("Database reset and system re-initialized.")
    
    st.subheader("System Logs")
    try:
        with open("travel_booking.log", "r") as f:
            logs = f.read()
        st.text_area("Logs", logs, height=300)
    except FileNotFoundError:
        st.info("No logs available yet.")


# ==================== APP ENTRY POINT ====================
if __name__ == "__main__":
    main()

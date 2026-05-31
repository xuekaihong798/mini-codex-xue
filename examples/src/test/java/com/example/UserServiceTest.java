package com.example;

import org.junit.Test;
import static org.junit.Assert.*;

public class UserServiceTest {

    @Test
    public void testValidChinesePhone() {
        assertTrue(UserService.isValidPhone("13812345678"));
        assertTrue(UserService.isValidPhone("15987654321"));
        assertTrue(UserService.isValidPhone("18800001111"));
    }

    @Test
    public void testInvalidPhone() {
        assertFalse(UserService.isValidPhone("12345678901")); // starts with 2
        assertFalse(UserService.isValidPhone("1381234567"));  // too short
        assertFalse(UserService.isValidPhone("138123456789")); // too long
        assertFalse(UserService.isValidPhone(""));
        assertFalse(UserService.isValidPhone(null));
    }

    @Test
    public void testFormatPhone() {
        assertEquals("138-1234-5678", UserService.formatPhone("13812345678"));
        assertNull(UserService.formatPhone("12345"));
    }

    @Test
    public void testNormalizePhone() {
        assertEquals("13812345678", UserService.normalizePhone("138-1234-5678"));
        assertEquals("13812345678", UserService.normalizePhone("(138) 1234 5678"));
    }
}

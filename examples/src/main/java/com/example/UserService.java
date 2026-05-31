package com.example;

/**
 * User service with phone number validation.
 */
public class UserService {

    /**
     * Validates a phone number for China (1[3-9]...) or Japan (0[789]0...).
     * China Rules: 11 digits, starts with 1[3-9].
     * Japan Rules: 11 digits, starts with 0[789]0.
     */
    public static boolean isValidPhone(String phone) {
        if (phone == null || phone.isBlank()) {
            return false;
        }
        // Chinese: ^1[3-9]\d{9}$
        // Japanese: ^0[789]0\d{8}$
        // Combined regex for 11 digits
        return phone.matches("^(1[3-9]\\d{9}|0[789]0\\d{8})$");
    }

    public static String formatPhone(String phone) {
        if (phone == null || !isValidPhone(phone)) {
            return null;
        }
        return phone.substring(0, 3) + "-" + phone.substring(3, 7) + "-" + phone.substring(7);
    }

    public static String normalizePhone(String phone) {
        if (phone == null) return null;
        return phone.replaceAll("[\\s\\-\\(\\)]", "");
    }
}

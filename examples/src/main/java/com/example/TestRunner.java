package com.example;

/**
 * Self-contained test runner for UserService.
 */
public class TestRunner {
    static int passed = 0;
    static int failed = 0;

    static void check(String name, boolean condition) {
        if (condition) {
            passed++;
        } else {
            failed++;
            System.err.println("  FAIL: " + name);
        }
    }

    public static void main(String[] args) {
        check("valid 138", UserService.isValidPhone("13812345678"));
        check("valid 159", UserService.isValidPhone("15987654321"));
        check("invalid starts 2", !UserService.isValidPhone("22345678901"));
        check("invalid too short", !UserService.isValidPhone("1381234567"));
        check("invalid empty", !UserService.isValidPhone(""));
        check("invalid null", !UserService.isValidPhone(null));
        check("format", "138-1234-5678".equals(UserService.formatPhone("13812345678")));
        check("normalize", "13812345678".equals(UserService.normalizePhone("138-1234-5678")));

        System.out.println("\n" + passed + "/" + (passed + failed) + " tests passed");
        System.exit(failed > 0 ? 1 : 0);
    }
}

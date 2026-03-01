// Auto-generated JUnit 5 tests for {{LIB_NAME}}
package com.anycompute;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class AnyComputeTest {

    @Test
    void createAndFree() {
        try (var src = new AnyCompute()) {
            assertNotNull(src);
        }
    }

    @Test
    void multipleCreateFree() {
        for (int i = 0; i < 100; i++) {
            try (var src = new AnyCompute()) {
                assertNotNull(src);
            }
        }
    }
}

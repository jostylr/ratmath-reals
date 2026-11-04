import { describe, expect, test } from "bun:test";
import { Rational } from "@ratmath/core";
import { EXP, LN, PI } from "../src/index.js";

const inside = (inner, outer) =>
    outer.low.compareTo(inner.low) <= 0 && outer.high.compareTo(inner.high) >= 0;
const containsDecimal = (interval, text) => {
    const value = new Rational(text);
    const unit = new Rational(1n, 10n ** BigInt(text.split(".")[1].length));
    return interval.low.compareTo(value.add(unit)) <= 0 && interval.high.compareTo(value) >= 0;
};

describe("transcendental results enclose the true value", () => {
    test("ln 2 and e at 40 digits", () => {
        expect(containsDecimal(LN(new Rational(2), -40), "0.6931471805599453094172321214581765680755")).toBe(true);
        expect(containsDecimal(EXP(new Rational(1), -40), "2.7182818284590452353602874713526624977572")).toBe(true);
        expect(containsDecimal(PI(-40), "3.1415926535897932384626433832795028841971")).toBe(true);
    });

    test("finer precision nests inside coarser precision", () => {
        for (const x of [new Rational(1), new Rational(-3, 2), new Rational(10), new Rational(100, 7), new Rational(1, 1000)]) {
            expect(inside(EXP(x, -40), EXP(x, -10))).toBe(true);
            expect(inside(LN(x.abs(), -40), LN(x.abs(), -10))).toBe(true);
        }
    });

    test("exp of a logarithm enclosure contains the argument", () => {
        for (const y of [new Rational(2), new Rational(5, 3), new Rational(1, 7), new Rational(1000)]) {
            const log = LN(y, -30);
            expect(EXP(log.low, -30).low.compareTo(y) <= 0).toBe(true);
            expect(EXP(log.high, -30).high.compareTo(y) >= 0).toBe(true);
        }
    });
});

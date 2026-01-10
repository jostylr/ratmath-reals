
import * as Reals from './index.js';

/**
 * RatMath Module Wrapper for @ratmath/reals
 * This file exports a structure compatible with VariableManager.loadModule
 */


// Helper to resolve precision from argument or environment
function getPrecision(context, prec) {
    if (prec !== undefined) return prec;
    if (context && context.variables) {
        let val;
        // Prioritize underscore prefixed variable as per new convention
        if (context.variables.has("_precision")) val = context.variables.get("_precision");
        else if (context.variables.has("PRECISION")) val = context.variables.get("PRECISION");

        if (val !== undefined) {
            // Convert Rational/Integer to number if needed, or pass as is if Reals supports it
            // Reals.js expects number or object with epsilon
            if (val && val.toNumber) return val.toNumber();
            return val;
        }
    }
    return undefined; // Reals will use default
}

export const functions = {
    "PI": {
        type: 'js',
        body: function (prec) { return Reals.PI(getPrecision(this, prec)); },
        params: ["precision?"],
        doc: "Returns PI with optional precision (default: 10^-6 or env PRECISION)"
    },
    "E": {
        type: 'js',
        body: function (prec) { return Reals.E(getPrecision(this, prec)); },
        params: ["precision?"],
        doc: "Returns e (Euler's number) with optional precision"
    },
    "Exp": {
        type: 'js',
        body: function (x, prec) { return Reals.EXP(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes e^x with optional precision"
    },
    "Ln": {
        type: 'js',
        body: function (x, prec) { return Reals.LN(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes natural logarithm of x"
    },
    "Log": {
        type: 'js',
        body: function (x, base, prec) { return Reals.LOG(x, base, getPrecision(this, prec)); },
        params: ["x", "base", "precision?"], // Base is mandatory here unless we make it optional too? User asked about optional basics.
        doc: "Computes logarithm of x base b"
    },
    "Sin": {
        type: 'js',
        body: function (x, prec) { return Reals.SIN(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes sine of x (radians)"
    },
    "Cos": {
        type: 'js',
        body: function (x, prec) { return Reals.COS(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes cosine of x (radians)"
    },
    "Tan": {
        type: 'js',
        body: function (x, prec) { return Reals.TAN(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes tangent of x (radians)"
    },
    "Arcsin": {
        type: 'js',
        body: function (x, prec) { return Reals.ARCSIN(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes inverse sine of x"
    },
    "Arccos": {
        type: 'js',
        body: function (x, prec) { return Reals.ARCCOS(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes inverse cosine of x"
    },
    "Arctan": {
        type: 'js',
        body: function (x, prec) { return Reals.ARCTAN(x, getPrecision(this, prec)); },
        params: ["x", "precision?"],
        doc: "Computes inverse tangent of x"
    },
    "Root": {
        type: 'js',
        body: function (q, n, prec) { return Reals.newtonRoot(q, n, getPrecision(this, prec)); },
        params: ["q", "n", "precision?"],
        doc: "Computes nth root of q using Newton's method"
    },
    "Pow": {
        type: 'js',
        body: function (base, exp, prec) { return Reals.rationalIntervalPower(base, exp, getPrecision(this, prec)); },
        params: ["base", "exponent", "precision?"],
        doc: "Computes base^exponent (supports fractional exponents)"
    }
};

export const variables = {
    // We could export constants here too if we wanted them as simple vars, but functions allow precision arg
};

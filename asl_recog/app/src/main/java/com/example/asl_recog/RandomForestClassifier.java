package com.example.asl_recog;

import java.util.ArrayList;
import java.util.List;

public class RandomForestClassifier {
    private List<DecisionTree> trees;
    private int numClasses;

    public RandomForestClassifier(List<DecisionTree> trees, int numClasses) {
        this.trees = trees;
        this.numClasses = numClasses;
    }

    public int predict(float[] features) {
        int[] classCounts = new int[numClasses];
        for (DecisionTree tree : trees) {
            int prediction = tree.predict(features);
            classCounts[prediction]++;
        }

        int maxCount = -1;
        int predictedClass = -1;
        for (int i = 0; i < numClasses; i++) {
            if (classCounts[i] > maxCount) {
                maxCount = classCounts[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }

    public static class DecisionTree {
        private Node root;

        public DecisionTree(Node root) {
            this.root = root;
        }

        public int predict(float[] features) {
            Node current = root;
            while (!current.isLeaf) {
                if (features[current.featureIndex] <= current.threshold) {
                    current = current.left;
                } else {
                    current = current.right;
                }
            }
            return current.prediction;
        }
    }

    public static class Node {
        boolean isLeaf;
        int featureIndex;
        float threshold;
        int prediction;
        Node left;
        Node right;

        // Constructor for leaf node
        public Node(int prediction) {
            this.isLeaf = true;
            this.prediction = prediction;
        }

        // Constructor for internal node
        public Node(int featureIndex, float threshold, Node left, Node right) {
            this.isLeaf = false;
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
    }
}
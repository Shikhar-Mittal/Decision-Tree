import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.lang.Math;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	// ordered list of class labels
	private List<String> labels;
	// ordered list of attributes
	private List<String> attributes;
	// map to ordered discrete values taken by attributes
	private Map<String, List<String>> attributeValues;
	// map for getting the index
	private HashMap<String, Integer> label_inv;
	private HashMap<String, Integer> attr_inv;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train:
	 *            the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: Homework requirement, learn the decision tree here
		// Get the list of instances via train.instances
		// You should write a recursive helper function to build the tree
		//
		// this.labels contains the possible labels for an instance
		// this.attributes contains the whole set of attribute names
		// train.instances contains the list of instances
		root = buildtree(attributes, train.instances, majorityLabel(train.instances), null);
	}

	// Helper method for Build Tree
	public DecTreeNode buildtree(List<String> attributes, List<Instance> train, String defaultLabel,
			String parentAttribute) {

		// if the example set is empty
		if (train.isEmpty()) {
			DecTreeNode newTree = new DecTreeNode(defaultLabel, "", parentAttribute, true);
			return newTree;
		}
		// if the examples have the same label y -> return y
		if (sameLabel(train)) {
			DecTreeNode newTree = new DecTreeNode(majorityLabel(train), "", parentAttribute, true);
			return newTree;
		}
		// if the attributes are empty -> return majority class of examples
		if (attributes.isEmpty()) {
			DecTreeNode newTree = new DecTreeNode(majorityLabel(train), "", parentAttribute, true);
			return newTree;
		}

		// defining best info gain as the lowest possible negative value
		double bestInfoGain = (-1) * Double.MAX_VALUE;
		String chosenAttribute = "";

		// counter for the best info gain assignment
		int bestCount = -1;

		// for loop to get the infogain from the attributes
		for (int i = 0; i < attributes.size(); i++) {
			double infoGain = (InfoGain(train, attributes.get(i)));

			// if the infogain is an improvement -> store it
			if (infoGain > bestInfoGain) {
				bestInfoGain = infoGain;
				bestCount = i;
			}
		}

		chosenAttribute = attributes.get(bestCount);

		DecTreeNode tree = new DecTreeNode(majorityLabel(train), chosenAttribute, parentAttribute, false);

		// Creating a list to take all the possible attribute values
		List<String> attributePossibilities = attributeValues.get(chosenAttribute);

		for (int i = 0; i < attributePossibilities.size(); i++) {

			int attributeIndex = getAttributeIndex(chosenAttribute);
			List<Instance> newExamples = new ArrayList<Instance>();

			// going through the training set to save the desired attributes
			for (int j = 0; j < train.size(); j++) {

				if (train.get(j).attributes.get(attributeIndex).equals(attributePossibilities.get(i)))
					newExamples.add(train.get(j));
			}

			// preserving old attributes
			List<String> newAttributes = new ArrayList<String>(attributes);

			// removing some attributes -> dodging stack overflow error
			newAttributes.remove(chosenAttribute);

			// generating tree through recursive call
			if (newExamples.size() == 0)
				tree.addChild(
						buildtree(newAttributes, newExamples, majorityLabel(train), attributePossibilities.get(i)));
			else
				tree.addChild(
						buildtree(newAttributes, newExamples, majorityLabel(train), attributePossibilities.get(i)));

		}

		// return the final tree
		return tree;
	}

	boolean sameLabel(List<Instance> train) {
		// Suggested helper function
		// returns if all the instances have the same label
		// labels are in instances.get(i).label

		// boolean for checking the presence of the label in the instance
		boolean labelCheck = true;

		for (int i = 0; i < train.size(); i++) {

			// checking if the training set contains the label
			if (!train.get(0).label.equals((train.get(i).label))) {
				labelCheck = false;
				break;
			}

		}
		return labelCheck;
	}

	String majorityLabel(List<Instance> train) {
		// Suggested helper function
		// returns the majority label of a list of examples

		// initializing the checker counters
		int positive = 0;
		int negative = 0;

		// for loop to iterate through the instances list
		for (int i = 0; i < train.size(); i++) {

			// if a hit -> increment positive
			if (train.get(i).label.equals(this.labels.get(0)))
				positive++;

			// else increment negative
			else
				negative++;
		}

		// returning the majority label for the training set
		if (positive >= negative)
			return this.labels.get(0);

		else
			return this.labels.get(1);

	}

	double entropy(List<Instance> train) {
		// Suggested helper function
		// returns the Entropy of a list of examples

		// initializing the good and bad counter to 0
		int g = 0;
		int b = 0;

		// if the training set is empty -> return 0
		if (train.size() == 0)
			return 0;

		// importing the instances' 0 index to a string
		String arbitraryLabel = train.get(0).label;

		// for loop to iterate through the entire traning set
		for (int i = 0; i < train.size(); i++) {

			// if there is a match -> increment good
			if (train.get(i).label.equals(arbitraryLabel))
				g++;
			// else increment bad
			else
				b++;
		}

		// initializing the fraction floats
		float frac1 = (float) g / train.size();
		float frac2 = (float) b / train.size();

		// return statements for the entropy values
		if (frac1 == 1)
			return -frac1 * (Math.log(frac1) / (Math.log(2)));

		else if (frac2 == 1)
			return -frac2 * (Math.log(frac2) / (Math.log(2)));

		else {
			double ret1 = (-frac1) * (Math.log(frac1) / (Math.log(2)));
			double ret2 = (-frac2) * (Math.log(frac2) / (Math.log(2)));
			return ret1 + ret2;
		}
	}

	double conditionalEntropy(List<Instance> train, String attr) {
		// Suggested helper function
		// returns the conditional entropy of a list of examples, given the
		// attribute attr

		// initializing a list to assign the attribute values based on the
		// attributes index
		List<String> vals = attributeValues.get(attributes.get(getAttributeIndex(attr)));
		double cndtlEntropy = 0;

		// for loop with another nested for loop to iterate through the size of
		// the entire list
		for (int i = 0; i < vals.size(); i++) {
			List<Instance> valList = new ArrayList<Instance>();

			for (int j = 0; j < train.size(); j++) {

				// if the loop is satisfied -> add the training set to the
				// valList
				if (train.get(j).attributes.get(getAttributeIndex(attr)).equals(vals.get(i))) {
					valList.add(train.get(j));
				}
			}
			// incrementing the conditional entropy as long as the valList is
			// not empty
			if (valList.size() != 0) {
				cndtlEntropy += entropy(valList) * ((double) (valList.size()) / (double) (train.size()));
			}
		}

		// return the conditional entropy
		return cndtlEntropy;
	}

	double InfoGain(List<Instance> train, String attr) {
		// Suggested helper function
		// returns the info gain of a list of examples, given the attribute attr
		return entropy(train) - conditionalEntropy(train, attr);
	}

	public String recursiveClassify(Instance train, DecTreeNode node) {
		// recursive method to call classify

		// null string label
		String label = null;

		// if we have a terminal node -> return the label
		if (node.terminal)
			return node.label;

		// else iterate through the node's children
		else {

			for (int i = 0; i < node.children.size(); i++) {
				// defining the 1st child in the decision tree
				DecTreeNode child1 = node.children.get(i);

				if (train.attributes.get(getAttributeIndex(node.attribute)).equals(child1.parentAttributeValue)) {
					// calling the recursive classify method
					label = recursiveClassify(train, child1);
					break;
				}
			}
		}
		// return the String label
		return label;
	}

	@Override
	public String classify(Instance train) {
		// The tree is already built, when this function is called
		// this.root will contain the learned decision tree.
		// write a recursive helper function, to return the predicted label of
		// instance

		// initializing a string and calling the recursive classify method
		String labelOfInstance = null;
		labelOfInstance = recursiveClassify(train, root);

		// return the predicted good or bad value of the training set instance
		return labelOfInstance;
	}

	@Override
	public void rootInfoGain(DataSet train) {
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: Homework requirement
		// Print the Info Gain for using each attribute at the root node
		// The decision tree may not exist when this function is called.
		// But you just need to calculate the info gain with each attribute,
		// on the entire training set.
		for (int i = 0; i < attributes.size(); i++) {
			System.out.format("%s %.5f\n", attributes.get(i), InfoGain(train.instances, attributes.get(i)));
		}
	}

	@Override
	public void printAccuracy(DataSet test) {
		// Print the accuracy on the test set.
		// The tree is already built, when this function is called
		// You need to call function classify, and compare the predicted labels.
		// List of instances: test.instances
		// getting the real label: test.instances.get(i).label

		// initializing the misses
		int misses = 0;
		
		// for loop to iterate through the teest sample
		for (int i = 0; i < test.instances.size(); i++) {
			String label = classify(test.instances.get(i));
			// if the label doesn't match -> increment the misses
			if (!label.equals(test.instances.get(i).label)) 
				misses++;
		
		}

		float accuracy = ((float) test.instances.size() - (float) misses) / ((float) test.instances.size());
		// formatting the print accuracy 
		System.out.format("%.5f\n", accuracy);

		return;
	}

	@Override
	/**
	 * Print the decision tree in the specified format Do not modify
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node with each line prefixed by 4 * k spaces.
	 * Do not modify
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else {
			int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
			value = attributeValues.get(parent.attribute).get(attributeValueIndex);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + p.label + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + p.attribute + "?}");
			System.out.println(sb.toString());
			for (DecTreeNode child : p.children) {
				printTreeNode(child, p, k + 1);
			}
		}
	}

	/**
	 * Helper function to get the index of the label in labels list
	 */
	private int getLabelIndex(String label) {
		if (label_inv == null) {
			this.label_inv = new HashMap<String, Integer>();
			for (int i = 0; i < labels.size(); i++) {
				label_inv.put(labels.get(i), i);
			}
		}
		return label_inv.get(label);
	}

	/**
	 * Helper function to get the index of the attribute in attributes list
	 */
	private int getAttributeIndex(String attr) {
		if (attr_inv == null) {
			this.attr_inv = new HashMap<String, Integer>();
			for (int i = 0; i < attributes.size(); i++) {
				attr_inv.put(attributes.get(i), i);
			}
		}
		return attr_inv.get(attr);
	}

	/**
	 * Helper function to get the index of the attributeValue in the list for
	 * the attribute key in the attributeValues map
	 */
	private int getAttributeValueIndex(String attr, String value) {
		for (int i = 0; i < attributeValues.get(attr).size(); i++) {
			if (value.equals(attributeValues.get(attr).get(i))) {
				return i;
			}
		}
		return -1;
	}
}
